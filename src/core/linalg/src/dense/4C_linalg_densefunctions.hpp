// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_DENSEFUNCTIONS_HPP
#define FOUR_C_LINALG_DENSEFUNCTIONS_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_mathoperations.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg::DenseFunctions
{
  class SerialDenseMatrix;
  class SerialDenseVector;

  /*
   * Declaration of the functions taking value_type*
   *
   */

  /// Multiplication: \e out = \e left*\e right
  /*!
    Multiply \e left and \e right and store the result in \e out. This
    function takes three template parameters \c i, \c j and \c k denoting
    the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e left*\e right
  /*!
    Multiply \e left and \e right and store the result in \e out. This
    function takes three template parameters \c i, \c j and \c k denoting
    the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_nn(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e left*\e right^T
  /*!
    Multiply \e left and \e right^T and store the result in \e out. This
    function takes three template parameters \c i, \c j and \c k denoting
    the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_nt(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e left^T*\e right
  /*!
    Multiply \e left^T and \e right and store the result in \e out. This
    function takes three template parameters \c i, \c j and \c k denoting
    the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_tn(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e left^T*\e right^T
  /*!
    Multiply \e left^T and \e right^T and store the result in \e out. This
    function takes three template parameters \c i, \c j and \c k denoting
    the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_tt(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e infac * \e left*\e right
  /*!
    Multiply \e left and \e right, scale the result by \e infac and store
    it in \e out. This function takes three template parameters \c
    i, \c j and \c k denoting the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e infac * \e left*\e right
  /*!
    Multiply \e left and \e right, scale the result by \e infac and store
    it in \e out. This function takes three template parameters \c
    i, \c j and \c k denoting the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nn(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e infac * \e left*\e right^T
  /*!
    Multiply \e left and \e right^T, scale the result by \e infac and store
    it in \e out. This function takes three template parameters \c
    i, \c j and \c k denoting the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right^T
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nt(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e infac * \e left^T*\e right
  /*!
    Multiply \e left^T and \e right, scale the result by \e infac and store
    it in \e out. This function takes three template parameters \c
    i, \c j and \c k denoting the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left^T*right
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tn(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e infac * \e left^T*\e right^T
  /*!
    Multiply \e left^T and \e right^T, scale the result by \e infac and store
    it in \e out. This function takes three template parameters \c
    i, \c j and \c k denoting the sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left^T*right^T
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tt(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right);

  /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right
  /*!
    Scale \e out by \e outfac and add \e left*\e right scaled by \e
    infac. This function takes three template parameters \c i, \c j
    and \c k denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right);

  /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right
  /*!
    Scale \e out by \e outfac and add \e left*\e right scaled by \e
    infac. This function takes three template parameters \c i, \c j
    and \c k denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nn(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right);

  /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right^T
  /*!
    Scale \e out by \e outfac and add \e left*\e right^T scaled by \e
    infac. This function takes three template parameters \c i, \c j
    and \c k denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left*right^T
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nt(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right);

  /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left^T*\e right
  /*!
    Scale \e out by \e outfac and add \e left^T*\e right scaled by \e
    infac. This function takes three template parameters \c i, \c j
    and \c k denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left^T*right
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tn(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right);

  /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left^T*\e right^T
  /*!
    Scale \e out by \e outfac and add \e left^T*\e right^T scaled by \e
    infac. This function takes three template parameters \c i, \c j
    and \c k denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c k)
    \param infac
      scalar to multiply with \e left^T*right^T
    \param left
      pointer to the first factor, size (\c j)x(\c i) so that \e
      left^T has size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c k)x(\c j) so that \e
      right^T has size (\c j)x(\c k)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tt(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right);

  /// invert matrix: \e out = inv(\e in)
  /*!
    invert the matrix \e in and store the result in \e out. To keep a
    common interface there are two template parameters \c i and \c j, but
    they must be the same number. The sizes of \e in and \e out are
    expected to be (\c i)x(\c j), and they must be square.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c j)
    \param in
      pointer to the matrix to be inverted, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType invert(ValueType* out, const ValueType* in);

  /// invert matrix: \e mat = inv(\e mat)
  /*!
    invert the matrix \e mat in place. To keep a common interface there
    are two template parameters \c i and \c j, but they must be the same
    number. The size of \e mat is expected to be (\c i)x(\c j), and it must
    be square.

    \param mat
      pointer to the matrix to be inverted in place, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType invert(ValueType* mat);

  /// Compute determinant
  /*!
    Computes and returns the determinant of \e mat. To keep a common
    interface there are two template parameters \c i and \c j, but they
    must be the same number. The size of \e mat is expected to be
    (\c i)x(\c j), and it must be square.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return determinant
   */
  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType determinant(const ValueType* mat);

  /// Compute
  /*!
    Computes and returns the trace of \e mat. To keep a common
    interface there are two template parameters \c i and \c j, but they
    must be the same number. The size of \e mat is expected to be
    (\c i)x(\c j), and it must be square.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return determinant
   */
  template <unsigned int i, unsigned int j, class ValueType>
  constexpr inline ValueType trace(const ValueType* mat);

  /// Copy: \e out = \e in
  /*!
    Copy \e in to \e out. This function takes two template parameters \c i and
    \c j denoting the sizes of the matrices.

    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param in
      pointer to the matrix to be copied, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeIn>
  constexpr inline void update(ValueTypeOut* out, const ValueTypeIn* in);

  /// Scaled copy: \e out = \e infac * \e in
  /*!
    Scale \e in by \e infac and store the result in \e out. This function takes two template
    parameters \c i and \c j denoting the sizes of the matrices.

    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param infac
      scalar to multiply with \e in
    \param in
      pointer to the matrix to read from, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeInfac,
      class ValueTypeIn>
  constexpr inline void update(
      ValueTypeOut* out, const ValueTypeInfac infac, const ValueTypeIn* in);

  /// Addition: \e out = \e outfac * \e out + \e infac * \e in
  /*!
    Scale \e out by \e outfac and add \e infac * \e in to it. This function
    takes two template parameters \c i and \c j denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param infac
      scalar to multiply with \e in
    \param in
      pointer to the matrix to be added, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeInfac, class ValueTypeIn>
  constexpr inline void update(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeIn* in);

  /// Addition: \e out = \e left + \e right
  /*!
    Add \e left and \e right and store the result in \e out. This
    function takes two template parameters \c i and \c j denoting the
    sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c j)
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param right
      pointer to the second factor, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void update(
      ValueTypeOut* out, const ValueTypeLeft* left, const ValueTypeRight* right);

  /// Addition: \e out = \e leftfac * \e left + \e rightfac * \e right
  /*!
    Add \e left and \e right, scaled by \e leftfac and \e rightfac
    respectively. The result is stored in \e out. This
    function takes two template parameters \c i and \c j denoting the
    sizes of the matrices.

    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c j)
    \param leftfac
      scalar to multiply with \e left
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param rightfac
      scalar to multiply with \e right
    \param right
      pointer to the second factor, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeftfac,
      class ValueTypeLeft, class ValueTypeRightfac, class ValueTypeRight>
  constexpr inline void update(ValueTypeOut* out, const ValueTypeLeftfac leftfac,
      const ValueTypeLeft* left, const ValueTypeRightfac rightfac, const ValueTypeRight* right);

  /// Addition: \e out = \e outfac * \e out + \e leftfac * \e left + \e rightfac * \e right
  /*!
    Scale \e out by \e outfac and add \e left and \e right, scaled by \e leftfac and \e rightfac
    respectively. The result is stored in \e out. This
    function takes two template parameters \c i and \c j denoting the
    sizes of the matrices.

    \param outfac
      scalar to multiply \e out with
    \param out
      pointer to the memory the result should be stored in, size (\c i)x(\c j)
    \param leftfac
      scalar to multiply with \e left
    \param left
      pointer to the first factor, size (\c i)x(\c j)
    \param rightfac
      scalar to multiply with \e right
    \param right
      pointer to the second factor, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeLeftfac, class ValueTypeLeft, class ValueTypeRightfac, class ValueTypeRight>
  constexpr inline void update(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeLeftfac leftfac, const ValueTypeLeft* left, const ValueTypeRightfac rightfac,
      const ValueTypeRight* right);

  /// Transposed copy: \e out = \e in^T
  /*!
    Copy transposed \e in to \e out. This function takes two template parameters \c i and
    \c j denoting the sizes of the matrices.

    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param in
      pointer to the matrix to be copied, size (\c j)x(\c i)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeIn>
  constexpr inline void update_t(ValueTypeOut* out, const ValueTypeIn* in);

  /// Scaled transposed copy: \e out = \e infac * \e in^T
  /*!
    Scale \e in by \e infac and store the transposed result in \e out. This function takes two
    template parameters \c i and \c j denoting the sizes of the matrices.

    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param infac
      scalar to multiply with \e in
    \param in
      pointer to the matrix to read from, size (\c j)x(\c i)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeInfac,
      class ValueTypeIn>
  constexpr inline void update_t(
      ValueTypeOut* out, const ValueTypeInfac infac, const ValueTypeIn* in);

  /// Transposed addition: \e out = \e outfac * \e out + \e infac * \e in^T
  /*!
    Scale \e out by \e outfac and add \e infac * \e in^T to it. This function
    takes two template parameters \c i and \c j denoting the sizes of the matrices.

    \param outfac
      scalar to multiply with \e out
    \param out
      pointer to the result matrix, size (\c i)x(\c j)
    \param infac
      scalar to multiply with \e in
    \param in
      pointer to the matrix to be added, size (\c i)x(\c j)
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeInfac, class ValueTypeIn>
  constexpr inline void update_t(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeIn* in);

  /// Multiply element-wise, \e out(m,n) = \e out(m,n)*\e in(m,n)
  /*!
    Multiply \e out and \e in, storing the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to first factor and result, size (\c i)x(\c j)
    \param in
      pointer to second factor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(ValueType* out, const ValueType* in);

  /// Multiply element-wise, \e out(m,n) = \e fac*\e out(m,n)*\e in(m,n)
  /*!
    Multiply \e out and \e in, scale by \e fac and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param fac
      scaling factor for the product
    \param out
      pointer to first factor and result, size (\c i)x(\c j)
    \param in
      pointer to second factor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      const ValueType fac, ValueType* out, const ValueType* in);

  /// Multiply element-wise, \e out(m,n) = \e left(m,n)*\e right(m,n)
  /*!
    Multiply \e left and \e right and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to result, size (\c i)x(\c j)
    \param left
      pointer to first factor, size (\c i)x(\c j)
    \param right
      pointer to second factor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      ValueType* out, const ValueType* left, const ValueType* right);

  /// Multiply element-wise, \e out(m,n) = \e infac*\e left(m,n)*\e right(m,n)
  /*!
    Multiply \e left and \e right, scale by \e infac and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to result, size (\c i)x(\c j)
    \param infac
      scaling factor
    \param left
      pointer to first factor, size (\c i)x(\c j)
    \param right
       pointer to second factor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      ValueType* out, const ValueType infac, const ValueType* left, const ValueType* right);

  /// Multiply element-wise, \e out(m,n) = \e outfac*\e out(m,n) + \e infac*\e left(m,n)*\e
  /// right(m,n)
  /*!
    Multiply \e left and \e right, scale by \e infac and add the result to \e out, scaled by \e
    outfac. This function takes two template parameters, unsigned ints \c i and \c j denoting the
    size of the matrices. \param outfac scaling factor for \e out \param out pointer to result,
    size (\c i)x(\c j) \param infac scaling factor the product \param left pointer to first
    factor, size (\c i)x(\c j) \param right pointer to second factor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(const ValueType outfac, ValueType* out,
      const ValueType infac, const ValueType* left, const ValueType* right);

  /// Divide element-wise, \e out(m,n) = \e out(m,n)/\e in(m,n)
  /*!
    Divide \e out by \e in, storing the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to dividend and result, size (\c i)x(\c j)
    \param in
      pointer to divisor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(ValueType* out, const ValueType* in);

  /// Divide element-wise, \e out(m,n) = \e fac*\e out(m,n)/\e in(m,n)
  /*!
    Divide \e out by \e in, scale by \e fac and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param fac
      scaling factor for the product
    \param out
      pointer to dividend and result, size (\c i)x(\c j)
    \param in
      pointer to divisor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(
      const ValueType fac, ValueType* out, const ValueType* in);

  /// Divide element-wise, \e out(m,n) = \e left(m,n)/\e right(m,n)
  /*!
    Divide \e left by \e right and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to result, size (\c i)x(\c j)
    \param left
      pointer to dividend, size (\c i)x(\c j)
    \param right
      pointer to divisor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(
      ValueType* out, const ValueType* left, const ValueType* right);

  /// Divide element-wise, \e out(m,n) = \e infac*\e left(m,n)/\e right(m,n)
  /*!
    Divide \e left by \e right, scale by \e infac and store the result in \e out.
    This function takes two template parameters, unsigned ints \c i and \c j
    denoting the size of the matrices.
    \param out
      pointer to result, size (\c i)x(\c j)
    \param infac
      scaling factor
    \param left
      pointer to dividend, size (\c i)x(\c j)
    \param right
       pointer to divisor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(
      ValueType* out, const ValueType infac, const ValueType* left, const ValueType* right);

  /// Divide element-wise, \e out(m,n) = \e outfac*\e out(m,n) + \e infac*\e left(m,n)/\e
  /// right(m,n)
  /*!
    Divide \e left by \e right, scale by \e infac and add the result to \e out, scaled by \e
    outfac. This function takes two template parameters, unsigned ints \c i and \c j denoting the
    size of the matrices. \param outfac scaling factor for \e out \param out pointer to result,
    size (\c i)x(\c j) \param infac scaling factor the product \param left pointer to dividend,
    size (\c i)x(\c j) \param right pointer to divisor, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(const ValueType outfac, ValueType* out,
      const ValueType infac, const ValueType* left, const ValueType* right);

  /// Scale matrix
  /*!
    Scale \e mat by \e fac. This function takes
    two template parameters \c i and \c j denoting the size of \e mat.

    \param fac
      scalar to multiply with \e mat
    \param mat
      pointer to the matrix, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void scale_matrix(const ValueType factor, ValueType* mat);

  /// Dot product
  /*!
    Return dot product \e left and \e right. This function
    takes two template parameters \c i and \c j denoting the sizes of the matrices.

    \param left
      pointer to the first matrix, size (\c i)x(\c j)
    \param right
      pointer to the second matrix, size (\c i)x(\c j)
    \return dot product
   */
  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline ValueTypeOut dot(const ValueTypeLeft* left, const ValueTypeRight* right);

  /// Set matrix to zero
  /*!
    Set matrix \e mat to zero. This function takes two template
    parameters i and j denoting the size of the matrix.

    This is the same as \e put_scalar<\c i, \c j>(0.0, \e mat), but it should be faster.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void clear_matrix(ValueType* mat);

  /// Fill matrix with scalar value
  /*!
    Set every number in \e mat to \e scalar. This function takes two template
    parameters \c i and \c j denoting the size of the matrix.

    \param scalar
      scalar value to be set
    \param mat
      pointer to the matrix, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void put_scalar(const ValueType scalar, ValueType* mat);

  /// Calculate absolute values of a matrix
  /*!
    Fill \e out with the absolute values from \e in. This function takes two
    template parameters \c i and \c j denoting the sizes of the matrices.

    \param out
      pointer to the matrix to be set, size (\c i)x(\c j)
    \param in
      pointer to the matrix the values are read from, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void abs(ValueType* out, const ValueType* in);

  /// Calculate reciprocal values of a matrix
  /*!
    Fill \e out with the reciprocal of the values from \e in. This
    function takes two template parameters \c i and \c j denoting the
    sizes of the matrices.

    \param out
      pointer to the matrix to be set, size (\c i)x(\c j)
    \param in
      pointer to the matrix the values are read from, size (\c i)x(\c j)
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void reciprocal(ValueType* out, const ValueType* in);

  /// 1-norm
  /*!
    This function computes the norm of the whole matrix. It returns
    a different result than Core::LinAlg::SerialDenseMatrix::OneNorm(),
    which returns the maximum of the norms of the columns.
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return 1-norm of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm1(const ValueType* mat);

  /// 2-norm (Euclidean norm)
  /*!
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return 2-norm of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm2(const ValueType* mat);

  /// Inf-norm
  /*!
    This function does not do the same as Core::LinAlg::SerialDenseMatrix::InfNorm().
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return inf-norm of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm_inf(const ValueType* mat);

  /// Minimum value of a matrix
  /*!
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return minimum value of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType min_value(const ValueType* mat);

  /// Maximum value of a matrix
  /*!
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return maximum value of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType max_value(const ValueType* mat);

  /// Mean value of a matrix
  /*!
    The template arguments \c i and \c j are the size of the matrix.

    \param mat
      pointer to the matrix, size (\c i)x(\c j)
    \return mean value of \e mat
   */
  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType mean_value(const ValueType* mat);


  /*
   * Definitions of the functions taking value_type*
   *
   */

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_nn(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_nt(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
        }
        *out = tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_tn(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3];
        }
        *out = tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void multiply_tt(
      ValueTypeOut* out, const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3 * k];
        }
        *out = tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nn(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nt(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
        }
        *out = infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tn(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3];
        }
        *out = infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tt(ValueTypeOut* out, const ValueTypeInfac infac,
      const ValueTypeLeft* const left, const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3 * k];
        }
        *out = infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = (*out) * outfac + infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nn(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3];
        }
        *out = (*out) * outfac + infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_nt(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i; ++c2)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
        }
        *out = (*out) * outfac + infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tn(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < j * k; c1 += j)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3];
        }
        *out = (*out) * outfac + infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
      class ValueTypeOutfac, class ValueTypeInfac, class ValueTypeLeft, class ValueTypeRight>
  constexpr inline void multiply_tt(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeLeft* const left,
      const ValueTypeRight* const right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    for (unsigned int c1 = 0; c1 < k; ++c1)
    {
      for (unsigned int c2 = 0; c2 < i * j; c2 += j)
      {
        ValueTypeOut tmp = left[c2] * right[c1];
        for (unsigned int c3 = 1; c3 < j; ++c3)
        {
          tmp += left[c2 + c3] * right[c1 + c3 * k];
        }
        *out = (*out) * outfac + infac * tmp;
        ++out;
      }
    }
  }

  template <class ValueType>
  inline ValueType invert1x1(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    const ValueType det = in[0];
    if (det == 0.0) FOUR_C_THROW("determinant of 1x1 matrix is zero");
    out[0] = 1.0 / in[0];
    return det;
  }

  template <class ValueType>
  inline ValueType invert2x2(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    const ValueType det = in[0] * in[1 + 1 * 2] - in[1] * in[1 * 2];
    if (det == 0.0) FOUR_C_THROW("determinant of 2x2 matrix is zero");
    const ValueType invdet = 1.0 / det;
    out[0] = invdet * in[1 + 1 * 2];
    out[1] = -invdet * in[1];
    out[1 * 2] = -invdet * in[1 * 2];
    out[1 + 1 * 2] = invdet * in[0];
    return det;
  }


  template <class ValueType>
  inline ValueType invert3x3(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    out[0] = in[1 + 1 * 3] * in[2 + 2 * 3] - in[2 + 1 * 3] * in[1 + 2 * 3];
    out[1] = in[2] * in[1 + 2 * 3] - in[1] * in[2 + 2 * 3];
    out[2] = in[1] * in[2 + 1 * 3] - in[2] * in[1 + 1 * 3];
    const ValueType det = in[0] * out[0] + in[1 * 3] * out[1] + in[2 * 3] * out[2];
    // const value_type det = in[0]*in[1+3*1]*in[2+3*2] +
    //                   in[0+3*1]*in[1+3*2]*in[2+3*0] +
    //                   in[0+3*2]*in[1+3*0]*in[2+3*1] -
    //                   in[0+3*2]*in[1+3*1]*in[2+3*0] -
    //                   in[0+3*0]*in[1+3*2]*in[2+3*1] -
    //                   in[0+3*1]*in[1+3*0]*in[2+3*2];
    if (det == 0.0) FOUR_C_THROW("determinant of 3x3 matrix is zero");
    const ValueType invdet = 1.0 / det;
    out[0] *= invdet;
    out[1] *= invdet;
    out[2] *= invdet;
    out[1 * 3] = invdet * (in[2 + 1 * 3] * in[2 * 3] - in[1 * 3] * in[2 + 2 * 3]);
    out[1 + 1 * 3] = invdet * (in[0] * in[2 + 2 * 3] - in[2] * in[2 * 3]);
    out[2 + 1 * 3] = invdet * (in[2] * in[1 * 3] - in[0] * in[2 + 1 * 3]);
    out[2 * 3] = invdet * (in[1 * 3] * in[1 + 2 * 3] - in[1 + 1 * 3] * in[2 * 3]);
    out[1 + 2 * 3] = invdet * (in[1] * in[2 * 3] - in[0] * in[1 + 2 * 3]);
    out[2 + 2 * 3] = invdet * (in[0] * in[1 + 1 * 3] - in[1] * in[1 * 3]);
    return det;
  }

  template <class ValueType>
  inline ValueType invert4x4(ValueType* out, const ValueType* in)
  {
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");

    const ValueType a00 = in[0];
    const ValueType a10 = in[1];
    const ValueType a20 = in[2];
    const ValueType a30 = in[3];
    const ValueType a01 = in[4];
    const ValueType a11 = in[5];
    const ValueType a21 = in[6];
    const ValueType a31 = in[7];
    const ValueType a02 = in[8];
    const ValueType a12 = in[9];
    const ValueType a22 = in[10];
    const ValueType a32 = in[11];
    const ValueType a03 = in[12];
    const ValueType a13 = in[13];
    const ValueType a23 = in[14];
    const ValueType a33 = in[15];

    // Compute determinant and check for invertibility
    const ValueType det = a00 * out[0] + a01 * out[1] + a02 * out[2] + a03 * out[3];
    if (det == 0.0) FOUR_C_THROW("determinant of 4x4 matrix is zero");

    // Compute adjugate matrix elements
    out[0] = a11 * a33 * a22 - a11 * a32 * a23 - a31 * a13 * a22 + a32 * a21 * a13 +
             a31 * a12 * a23 - a33 * a21 * a12;
    out[4] = -(a31 * a02 * a23 + a33 * a22 * a01 - a33 * a21 * a02 - a31 * a03 * a22 -
               a32 * a23 * a01 + a32 * a21 * a03);
    out[8] = -a11 * a33 * a02 + a11 * a32 * a03 + a33 * a01 * a12 + a31 * a13 * a02 -
             a32 * a01 * a13 - a31 * a12 * a03;
    out[12] = -(-a11 * a02 * a23 + a11 * a03 * a22 - a03 * a21 * a12 - a01 * a13 * a22 +
                a02 * a21 * a13 + a01 * a12 * a23);

    out[1] = -(a33 * a22 * a10 - a32 * a23 * a10 + a30 * a12 * a23 - a33 * a20 * a12 -
               a30 * a13 * a22 + a32 * a20 * a13);
    out[5] = a00 * a33 * a22 - a00 * a32 * a23 + a30 * a02 * a23 - a33 * a20 * a02 +
             a32 * a20 * a03 - a30 * a03 * a22;
    out[9] = -(a00 * a33 * a12 - a00 * a32 * a13 - a33 * a10 * a02 - a30 * a03 * a12 +
               a32 * a10 * a03 + a30 * a02 * a13);
    out[13] = -a00 * a13 * a22 + a00 * a12 * a23 - a10 * a02 * a23 + a13 * a20 * a02 +
              a10 * a03 * a22 - a12 * a20 * a03;

    out[2] = a11 * a30 * a23 - a11 * a33 * a20 - a31 * a10 * a23 + a33 * a21 * a10 +
             a31 * a13 * a20 - a30 * a21 * a13;
    out[6] = -(-a00 * a31 * a23 + a00 * a33 * a21 - a30 * a03 * a21 + a31 * a20 * a03 +
               a30 * a01 * a23 - a33 * a20 * a01);
    out[10] = a33 * a00 * a11 - a33 * a10 * a01 - a30 * a03 * a11 - a31 * a00 * a13 +
              a31 * a10 * a03 + a30 * a01 * a13;
    out[14] = -(a23 * a00 * a11 - a23 * a10 * a01 - a20 * a03 * a11 - a21 * a00 * a13 +
                a21 * a10 * a03 + a20 * a01 * a13);

    out[3] = -(a11 * a30 * a22 - a11 * a32 * a20 - a30 * a21 * a12 - a31 * a10 * a22 +
               a32 * a21 * a10 + a31 * a12 * a20);
    out[7] = -a00 * a31 * a22 + a00 * a32 * a21 + a31 * a20 * a02 + a30 * a01 * a22 -
             a32 * a20 * a01 - a30 * a02 * a21;
    out[11] = -(a32 * a00 * a11 - a32 * a10 * a01 - a30 * a02 * a11 - a31 * a00 * a12 +
                a31 * a10 * a02 + a30 * a01 * a12);
    out[15] = a22 * a00 * a11 - a22 * a10 * a01 - a20 * a02 * a11 - a21 * a00 * a12 +
              a21 * a10 * a02 + a20 * a01 * a12;

    // Compute the inverse by multiplying the adjugate by 1/det
    const ValueType invdet = 1.0 / det;
    for (int c = 0; c < 16; ++c) out[c] *= invdet;
    return det;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType invert(ValueType* out, const ValueType* in)
  {
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");

    static_assert(i == j, "Cannot compute inverse of non-square matrix");

    switch (i)
    {
      case 1:
        return invert1x1(out, in);
      case 2:
        return invert2x2(out, in);
      case 3:
        return invert3x3(out, in);
      case 4:
        return invert4x4(out, in);
      default:
        static_assert(i <= 4, "Cannot compute inverse of matrix bigger than 4x4");
        return 0.0;
    }
  }


  template <class ValueType>
  inline ValueType invert1x1(ValueType* mat)
  {
    const ValueType det = mat[0];
    if (det == 0.0) FOUR_C_THROW("determinant of 1x1 matrix is zero");
    mat[0] = 1.0 / mat[0];
    return det;
  }

  template <class ValueType>
  inline ValueType invert2x2(ValueType* mat)
  {
    ValueType tmp;
    const ValueType det = mat[0] * mat[1 + 1 * 2] - mat[1] * mat[1 * 2];
    if (det == 0.0) FOUR_C_THROW("determinant of 2x2 matrix is zero");
    const ValueType invdet = 1.0 / det;
    tmp = mat[0];
    mat[0] = invdet * mat[1 + 1 * 2];
    mat[1 + 1 * 2] = invdet * tmp;
    mat[1] *= -invdet;
    mat[1 * 2] *= -invdet;
    return det;
  }


  template <class ValueType>
  inline ValueType invert3x3(ValueType* mat)
  {
    const ValueType tmp00 = mat[1 + 1 * 3] * mat[2 + 2 * 3] - mat[2 + 1 * 3] * mat[1 + 2 * 3];
    const ValueType tmp10 = mat[2] * mat[1 + 2 * 3] - mat[1] * mat[2 + 2 * 3];
    const ValueType tmp20 = mat[1] * mat[2 + 1 * 3] - mat[2] * mat[1 + 1 * 3];
    const ValueType det = mat[0] * tmp00 + mat[1 * 3] * tmp10 + mat[2 * 3] * tmp20;
    // const value_type det = mat[0+3*0]*mat[1+3*1]*mat[2+3*2] +
    //                    mat[0+3*1]*mat[1+3*2]*mat[2+3*0] +
    //                    mat[0+3*2]*mat[1+3*0]*mat[2+3*1] -
    //                    mat[0+3*2]*mat[1+3*1]*mat[2+3*0] -
    //                    mat[0+3*0]*mat[1+3*2]*mat[2+3*1] -
    //                    mat[0+3*1]*mat[1+3*0]*mat[2+3*2];
    if (det == 0.0) FOUR_C_THROW("determinant of 3x3 matrix is zero");
    const ValueType invdet = 1.0 / det;
    const ValueType tmp01 = mat[1 * 3];
    const ValueType tmp11 = mat[1 + 1 * 3];
    const ValueType tmp12 = mat[1 + 2 * 3];
    mat[1 * 3] = invdet * (mat[2 + 1 * 3] * mat[2 * 3] - tmp01 * mat[2 + 2 * 3]);
    mat[1 + 1 * 3] = invdet * (mat[0] * mat[2 + 2 * 3] - mat[2] * mat[2 * 3]);
    mat[1 + 2 * 3] = invdet * (mat[1] * mat[2 * 3] - mat[0] * tmp12);
    mat[2 + 1 * 3] = invdet * (mat[2] * tmp01 - mat[0] * mat[2 + 1 * 3]);
    mat[2 * 3] = invdet * (tmp01 * tmp12 - tmp11 * mat[2 * 3]);
    mat[2 + 2 * 3] = invdet * (mat[0] * tmp11 - mat[1] * tmp01);
    mat[0] = invdet * tmp00;
    mat[1] = invdet * tmp10;
    mat[2] = invdet * tmp20;
    return det;
  }

  template <class ValueType>
  inline ValueType invert4x4(ValueType* mat)
  {
    const ValueType a00 = mat[0];
    const ValueType a10 = mat[1];
    const ValueType a20 = mat[2];
    const ValueType a30 = mat[3];
    const ValueType a01 = mat[4];
    const ValueType a11 = mat[5];
    const ValueType a21 = mat[6];
    const ValueType a31 = mat[7];
    const ValueType a02 = mat[8];
    const ValueType a12 = mat[9];
    const ValueType a22 = mat[10];
    const ValueType a32 = mat[11];
    const ValueType a03 = mat[12];
    const ValueType a13 = mat[13];
    const ValueType a23 = mat[14];
    const ValueType a33 = mat[15];

    // Compute determinant and check for invertibility
    const ValueType det = a00 * mat[0] + a01 * mat[1] + a02 * mat[2] + a03 * mat[3];
    if (det == 0.0) FOUR_C_THROW("determinant of 4x4 matrix is zero");

    // Compute adjugate matrix elements
    mat[0] = a11 * a33 * a22 - a11 * a32 * a23 - a31 * a13 * a22 + a32 * a21 * a13 +
             a31 * a12 * a23 - a33 * a21 * a12;
    mat[4] = -(a31 * a02 * a23 + a33 * a22 * a01 - a33 * a21 * a02 - a31 * a03 * a22 -
               a32 * a23 * a01 + a32 * a21 * a03);
    mat[8] = -a11 * a33 * a02 + a11 * a32 * a03 + a33 * a01 * a12 + a31 * a13 * a02 -
             a32 * a01 * a13 - a31 * a12 * a03;
    mat[12] = -(-a11 * a02 * a23 + a11 * a03 * a22 - a03 * a21 * a12 - a01 * a13 * a22 +
                a02 * a21 * a13 + a01 * a12 * a23);

    mat[1] = -(a33 * a22 * a10 - a32 * a23 * a10 + a30 * a12 * a23 - a33 * a20 * a12 -
               a30 * a13 * a22 + a32 * a20 * a13);
    mat[5] = a00 * a33 * a22 - a00 * a32 * a23 + a30 * a02 * a23 - a33 * a20 * a02 +
             a32 * a20 * a03 - a30 * a03 * a22;
    mat[9] = -(a00 * a33 * a12 - a00 * a32 * a13 - a33 * a10 * a02 - a30 * a03 * a12 +
               a32 * a10 * a03 + a30 * a02 * a13);
    mat[13] = -a00 * a13 * a22 + a00 * a12 * a23 - a10 * a02 * a23 + a13 * a20 * a02 +
              a10 * a03 * a22 - a12 * a20 * a03;

    mat[2] = a11 * a30 * a23 - a11 * a33 * a20 - a31 * a10 * a23 + a33 * a21 * a10 +
             a31 * a13 * a20 - a30 * a21 * a13;
    mat[6] = -(-a00 * a31 * a23 + a00 * a33 * a21 - a30 * a03 * a21 + a31 * a20 * a03 +
               a30 * a01 * a23 - a33 * a20 * a01);
    mat[10] = a33 * a00 * a11 - a33 * a10 * a01 - a30 * a03 * a11 - a31 * a00 * a13 +
              a31 * a10 * a03 + a30 * a01 * a13;
    mat[14] = -(a23 * a00 * a11 - a23 * a10 * a01 - a20 * a03 * a11 - a21 * a00 * a13 +
                a21 * a10 * a03 + a20 * a01 * a13);

    mat[3] = -(a11 * a30 * a22 - a11 * a32 * a20 - a30 * a21 * a12 - a31 * a10 * a22 +
               a32 * a21 * a10 + a31 * a12 * a20);
    mat[7] = -a00 * a31 * a22 + a00 * a32 * a21 + a31 * a20 * a02 + a30 * a01 * a22 -
             a32 * a20 * a01 - a30 * a02 * a21;
    mat[11] = -(a32 * a00 * a11 - a32 * a10 * a01 - a30 * a02 * a11 - a31 * a00 * a12 +
                a31 * a10 * a02 + a30 * a01 * a12);
    mat[15] = a22 * a00 * a11 - a22 * a10 * a01 - a20 * a02 * a11 - a21 * a00 * a12 +
              a21 * a10 * a02 + a20 * a01 * a12;

    // Compute the inverse by multiplying the adjugate by 1/det
    const ValueType invdet = 1.0 / det;
    for (int c = 0; c < 16; ++c) mat[c] *= invdet;
    return det;
  }


  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType invert(ValueType* mat)
  {
    static_assert(i == j, "Cannot compute inverse of non-square matrix");

    switch (i)
    {
      case 1:
        return invert1x1(mat);
      case 2:
        return invert2x2(mat);
      case 3:
        return invert3x3(mat);
      case 4:
        return invert4x4(mat);
      default:
        static_assert(i <= 4, "Cannot compute inverse of matrix bigger than 4x4");
        return 0.0;
    }
  }

  template <class ValueType>
  inline ValueType determinant_large_matrix(unsigned int i, unsigned int j, const ValueType* mat)
  {
    FOUR_C_THROW("determinant_large_matrix not implemented for this ValueType!");
  }

  // specialization for double using LAPACK
  double determinant_large_matrix(unsigned int i, unsigned int j, const double* mat);

  template <class ValueType, unsigned int i, unsigned int j>
  inline ValueType determinant(const ValueType* mat)
  {
    static_assert(i == j, "Matrix must be square");

    if constexpr (i == 1) return *mat;
    if constexpr (i == 2) return mat[0] * mat[1 + 1 * 2] - mat[1] * mat[1 * 2];
    if constexpr (i == 3)
    {
      return mat[0] * (mat[1 + 1 * 3] * mat[2 + 2 * 3] - mat[2 + 1 * 3] * mat[1 + 2 * 3]) +
             mat[1 * 3] * (mat[2] * mat[1 + 2 * 3] - mat[1] * mat[2 + 2 * 3]) +
             mat[2 * 3] * (mat[1] * mat[2 + 1 * 3] - mat[2] * mat[1 + 1 * 3]);
    }
    if constexpr (i == 4)
    {
      const ValueType a00 = mat[0];
      const ValueType a10 = mat[1];
      const ValueType a20 = mat[2];
      const ValueType a30 = mat[3];
      const ValueType a01 = mat[4];
      const ValueType a11 = mat[5];
      const ValueType a21 = mat[6];
      const ValueType a31 = mat[7];
      const ValueType a02 = mat[8];
      const ValueType a12 = mat[9];
      const ValueType a22 = mat[10];
      const ValueType a32 = mat[11];
      const ValueType a03 = mat[12];
      const ValueType a13 = mat[13];
      const ValueType a23 = mat[14];
      const ValueType a33 = mat[15];

      return a00 * a11 * a33 * a22 - a00 * a11 * a32 * a23 - a00 * a31 * a13 * a22 +
             a00 * a32 * a21 * a13 + a00 * a31 * a12 * a23 - a00 * a33 * a21 * a12 +
             a11 * a30 * a02 * a23 - a11 * a33 * a20 * a02 + a11 * a32 * a20 * a03 -
             a11 * a30 * a03 * a22 - a31 * a10 * a02 * a23 - a33 * a22 * a10 * a01 +
             a33 * a21 * a10 * a02 + a33 * a20 * a01 * a12 + a30 * a03 * a21 * a12 +
             a31 * a13 * a20 * a02 + a31 * a10 * a03 * a22 + a30 * a01 * a13 * a22 +
             a32 * a23 * a10 * a01 - a32 * a21 * a10 * a03 - a32 * a20 * a01 * a13 -
             a30 * a02 * a21 * a13 - a31 * a12 * a20 * a03 - a30 * a01 * a12 * a23;
    }

    return determinant_large_matrix(i, j, mat);
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType trace(const ValueType* mat)
  {
    static_assert(i == j, "Matrix must be square");
    static_assert(i > 0, "0x0 matrices are not supported for the trace");

    ValueType trace = *mat;
    for (unsigned int c = 1; c < i; ++c)
    {
      mat += i + 1;
      trace += *mat;
    }

    return trace;
  }


  /* add matrices */

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeIn>
  constexpr inline void update(ValueTypeOut* out, const ValueTypeIn* in)
  {
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
      // std::memcpy(out, in, i*j*sizeof(value_type));
      std::copy(in, in + i * j, out);
    }
    else
    {
      update<ValueTypeOut, i, j>(out, 1.0, in);
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeInfac,
      class ValueTypeIn>
  constexpr inline void update(ValueTypeOut* out, const ValueTypeInfac infac, const ValueTypeIn* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out = infac * (*in);
    for (unsigned int c = 1; c < i * j; ++c) *(++out) = infac * (*(++in));
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeInfac, class ValueTypeIn>
  constexpr inline void update(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeIn* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    if (outfac > -1e-30 and outfac < 1e-30)
    {  // cannot handle this case here, because 0*nan==nan
      update<ValueTypeOut, i, j>(out, infac, in);
      return;
    }
    *out *= outfac;
    *out += infac * (*in);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      *(++out) *= outfac;
      *out += infac * (*(++in));
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void update(
      ValueTypeOut* out, const ValueTypeLeft* left, const ValueTypeRight* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = *left + *right;
    for (unsigned int c = 1; c < i * j; ++c) *(++out) = *(++left) + *(++right);
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeftfac,
      class ValueTypeLeft, class ValueTypeRightfac, class ValueTypeRight>
  constexpr inline void update(ValueTypeOut* out, const ValueTypeLeftfac leftfac,
      const ValueTypeLeft* left, const ValueTypeRightfac rightfac, const ValueTypeRight* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = leftfac * (*left) + rightfac * (*right);
    for (unsigned int c = 1; c < i * j; ++c)
      *(++out) = leftfac * (*(++left)) + rightfac * (*(++right));
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeLeftfac, class ValueTypeLeft, class ValueTypeRightfac, class ValueTypeRight>
  constexpr inline void update(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeLeftfac leftfac, const ValueTypeLeft* left, const ValueTypeRightfac rightfac,
      const ValueTypeRight* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeLeft>)
      FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeRight>)
      FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    if (outfac > -1e-30 and outfac < 1e-30)
    {  // cannot handle this case here, because 0*nan==nan
      update<ValueTypeOut, i, j>(out, leftfac, left, rightfac, right);
      return;
    }
    *out *= outfac;
    *out += leftfac * (*left) + rightfac * (*right);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      *(++out) *= outfac;
      *out += leftfac * (*(++left)) + rightfac * (*(++right));
    }
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeIn>
  constexpr inline void update_t(ValueTypeOut* out, const ValueTypeIn* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    for (unsigned int c2 = 0; c2 < j; c2 += 1)
      for (unsigned int c1 = 0; c1 < i; c1 += 1) *(out++) = in[c2 + c1 * j];
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeInfac,
      class ValueTypeIn>
  constexpr inline void update_t(
      ValueTypeOut* out, const ValueTypeInfac infac, const ValueTypeIn* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    for (unsigned int c2 = 0; c2 < j; c2 += 1)
      for (unsigned int c1 = 0; c1 < i; c1 += 1) *(out++) = infac * in[c2 + c1 * j];
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeOutfac,
      class ValueTypeInfac, class ValueTypeIn>
  constexpr inline void update_t(const ValueTypeOutfac outfac, ValueTypeOut* out,
      const ValueTypeInfac infac, const ValueTypeIn* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if constexpr (std::is_same_v<ValueTypeOut, ValueTypeIn>)
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    if (outfac > -1e-30 and outfac < 1e-30)
    {  // cannot handle this case here, because 0*nan==nan
      update_t<ValueTypeOut, i, j>(out, infac, in);
      return;
    }
    for (unsigned int c2 = 0; c2 < j; c2 += 1)
    {
      for (unsigned int c1 = 0; c1 < i; c1 += 1)
      {
        *(out) *= outfac;
        *(out++) += infac * in[c2 + c1 * j];
      }
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out *= *in;
    for (unsigned c = 1; c < i * j; ++c) *(++out) *= *(++in);
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      const ValueType fac, ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out *= fac * (*in);
    for (unsigned c = 1; c < i * j; ++c) *(++out) *= fac * (*(++in));
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      ValueType* out, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = (*left) * (*right);
    for (unsigned c = 1; c < i * j; ++c) *(++out) = (*(++left)) * (*(++right));
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(
      ValueType* out, const ValueType infac, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = infac * (*left) * (*right);
    for (unsigned c = 1; c < i * j; ++c) *(++out) = infac * (*(++left)) * (*(++right));
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_multiply(const ValueType outfac, ValueType* out,
      const ValueType infac, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    if (outfac > -1e-30 and outfac < 1e-30)
    {
      elementwise_multiply<ValueType, i, j>(out, infac, left, right);
      return;
    }
    *out = outfac * (*out) + infac * (*left) * (*right);
    for (unsigned c = 1; c < i * j; ++c)
    {
      ++out;
      *out = outfac * (*out) + infac * (*(++left)) * (*(++right));
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out /= *in;
    for (unsigned c = 1; c < i * j; ++c) *(++out) /= *(++in);
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(const ValueType fac, ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out = fac * (*out) / (*in);
    for (unsigned c = 1; c < i * j; ++c)
    {
      ++out;
      ++in;
      *out = fac * (*out) / (*in);
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(
      ValueType* out, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = (*left) / (*right);
    for (unsigned c = 1; c < i * j; ++c) *(++out) = (*(++left)) / (*(++right));
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(
      ValueType* out, const ValueType infac, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    *out = infac * (*left) / (*right);
    for (unsigned c = 1; c < i * j; ++c) *(++out) = infac * (*(++left)) / (*(++right));
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void elementwise_divide(const ValueType outfac, ValueType* out,
      const ValueType infac, const ValueType* left, const ValueType* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != left, "'out' and 'left' point to same memory location");
    FOUR_C_ASSERT(out != right, "'out' and 'right' point to same memory location");
#endif
    if (outfac > -1e-30 and outfac < 1e-30)
    {
      elementwise_divide<ValueType, i, j>(out, infac, left, right);
      return;
    }
    *out = outfac * (*out) + infac * (*left) / (*right);
    for (unsigned c = 1; c < i * j; ++c)
    {
      ++out;
      *out = outfac * (*out) + infac * (*(++left)) / (*(++right));
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void scale_matrix(const ValueType factor, ValueType* mat)
  {
    *mat *= factor;
    for (unsigned int c = 1; c < i * j; ++c) *(++mat) *= factor;
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline ValueTypeOut dot(const ValueTypeLeft* left, const ValueTypeRight* right)
  {
    ValueTypeOut res = (*left) * (*right);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++left;
      ++right;
      res += (*left) * (*right);
    }
    return res;
  }

  template <class ValueTypeOut, unsigned int i, unsigned int j, class ValueTypeLeft,
      class ValueTypeRight>
  constexpr inline void crossproduct(
      ValueTypeOut* out, const ValueTypeLeft* left, const ValueTypeRight* right)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (i != 3 || j != 1) FOUR_C_THROW("cross product only for 3x1 matrices available");
#endif
    out[0] = left[1] * right[2] - left[2] * right[1];
    out[1] = left[2] * right[0] - left[0] * right[2];
    out[2] = left[0] * right[1] - left[1] * right[0];
    return;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void clear_matrix(ValueType* mat)
  {
    // the memset method is needed for arbitrary precision (cln) data types instead of the fill
    // method std::memset(mat,0,i*j*sizeof(value_type));
    std::fill(mat, mat + i * j, 0.0);
  }


  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void put_scalar(const ValueType scalar, ValueType* mat)
  {
    *mat = scalar;
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      *mat = scalar;
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void abs(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out = *in >= 0 ? *in : -*in;
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++out;
      ++in;
      *out = *in >= 0 ? *in : -*in;
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void reciprocal(ValueType* out, const ValueType* in)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
    *out = 1.0 / (*in);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++out;
      ++in;
      *out = 1.0 / (*in);
    }
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm1(const ValueType* mat)
  {
    ValueType result = *mat >= 0 ? *mat : -(*mat);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      result += *mat >= 0 ? *mat : -(*mat);
    }
    return result;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm2(const ValueType* mat)
  {
    ValueType result = (*mat) * (*mat);
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      result += (*mat) * (*mat);
    }
    return Core::MathOperations<ValueType>::sqrt(result);
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType norm_inf(const ValueType* mat)
  {
    ValueType result = Core::MathOperations<ValueType>::abs(*mat);
    ValueType tmp;
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      tmp = Core::MathOperations<ValueType>::abs(*mat);
      result = std::max(result, tmp);
    }
    return result;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType min_value(const ValueType* mat)
  {
    ValueType result = *mat;
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      if (*mat < result) result = *mat;
    }
    return result;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType max_value(const ValueType* mat)
  {
    ValueType result = *mat;
    for (unsigned int c = 1; c < i * j; ++c)
    {
      ++mat;
      if (*mat > result) result = *mat;
    }
    return result;
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline ValueType mean_value(const ValueType* mat)
  {
    ValueType result = *mat;
    for (unsigned int c = 1; c < i * j; ++c) result += *(++mat);
    return result / (i * j);
  }

  template <class ValueType, unsigned int i, unsigned int j>
  constexpr inline void normalize(ValueType* vector)
  {
    ValueType norm2 = DenseFunctions::norm2<ValueType, i, j>(vector);
    if (norm2 < 1e-10)
    {
      FOUR_C_THROW("Cannot normalize vector with norm2 close to zero");
    }

    DenseFunctions::scale_matrix<ValueType, i, j>(1.0 / norm2, vector);
  }
}  // namespace Core::LinAlg::DenseFunctions

FOUR_C_NAMESPACE_CLOSE

#endif
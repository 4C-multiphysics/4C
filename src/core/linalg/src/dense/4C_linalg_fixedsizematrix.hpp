// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP
#define FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_mathoperations.hpp"

#include <ostream>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseMatrix;
  class SerialDenseVector;

  namespace DenseFunctions
  {
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
    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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
    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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
    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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
    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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
    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueTypeOut, unsigned int i, unsigned int j, unsigned int k,
        class ValueTypeLeft, class ValueTypeRight>
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

    template <class ValueType, unsigned int i, unsigned int j>
    inline ValueType invert(ValueType* out, const ValueType* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      FOUR_C_ASSERT(out != in, "'out' and 'in' point to same memory location");
#endif
      static_assert(i == j, "Cannot compute inverse of non-square matrix");

      switch (i)
      {
        case 1:
          return invert1x1(out, in);
        case 2:
          return invert2x2(out, in);
        case 3:
          return invert3x3(out, in);
        default:
          static_assert(i < 4, "Cannot compute inverse of matrix bigger than 3x3");
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
        default:
          static_assert(i < 4, "Cannot compute inverse of matrix bigger than 3x3");
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
    constexpr inline void update(
        ValueTypeOut* out, const ValueTypeInfac infac, const ValueTypeIn* in)
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
    constexpr inline void elementwise_divide(
        const ValueType fac, ValueType* out, const ValueType* in)
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
      FOUR_C_ASSERT_ALWAYS(j == 1, "normalize() is only available for column vectors");

      ValueType norm2 = DenseFunctions::norm2<ValueType, i, j>(vector);
      if (norm2 < 1e-10)
      {
        FOUR_C_THROW("Cannot normalize vector with norm2 close to zero");
      }

      DenseFunctions::scale_matrix<ValueType, i, j>(1.0 / norm2, vector);
    }
  }  // namespace DenseFunctions

  namespace Internal
  {
    /**
     * Get access to the underlying values.
     *
     * @note This function is used to hide the details of the SerialDenseMatrix in the
     * implementation.
     */
    const double* values(const Core::LinAlg::SerialDenseMatrix& matrix);

    /**
     * Get access to the underlying values.
     *
     * @note This function is used to hide the details of the SerialDenseMatrix in the
     * implementation.
     */
    double* values(Core::LinAlg::SerialDenseMatrix& matrix);

    /**
     * Get access to the underlying values.
     *
     * @note This function is used to hide the details of the SerialDenseVector in the
     * implementation.
     */
    const double* values(const Core::LinAlg::SerialDenseVector& vector);

    /**
     * Get access to the underlying values.
     *
     * @note This function is used to hide the details of the SerialDenseVector in the
     * implementation.
     */
    double* values(Core::LinAlg::SerialDenseVector& vector);
  }  // namespace Internal

  enum class Initialization
  {
    zero,
    uninitialized
  };

  /// Serial dense matrix with templated dimensions
  /*!
    A serial dense matrix with templated dimensions that is supposed to
    be fast and lightweight. The default scalar type is double.
    The value_type-array is allocated on the stack (small sizes, up to 512
    bytes or on the heap (larger sizes) and stored in
    column-major order, just like in Core::LinAlg::SerialDenseMatrix.

    The interface is based on that of Core::LinAlg::SerialDenseMatrix and
    Core::LinAlg::MultiVector<double>. The whole View/Copy thing works a little
    different, though. See the appropriate functions for details.

    There is no operator[]. It behaves differently in
    Core::LinAlg::SerialDenseMatrix and Core::LinAlg::SerialDenseVector::Base, and is not
    needed in either of them.
   */
  template <unsigned int rows, unsigned int cols, class ValueType = double>
  class Matrix
  {
    static_assert(rows > 0, "Number of rows must be greater than zero!");
    static_assert(cols > 0, "Number of columns must be greater than zero!");

   private:
    /// threshold for when to allocate the memory instead of placing the matrix on the stack.
    /// set to 512 bytes (or 64 entries for double matrices).
    static constexpr bool allocatesmemory_ = rows * cols * sizeof(ValueType) > 512;

    /// the pointer holding the data
    ValueType* data_;

    /// for small sizes of the matrix, avoid expensive memory allocation by storing
    /// the matrix on the stack
    ValueType datafieldsmall_[allocatesmemory_ ? 1 : rows * cols];

    /// whether we are a view to some other matrix
    bool isview_;

    /// only in combination with isview_. Pure read access to the underlying data.
    bool isreadonly_;

   public:
    using scalar_type = ValueType;

    /// Default constructor
    /*!
      Constructs a new matrix and allocates the memory. If \e init is set to set_zero, the matrix is
      initialized to zero, otherwise it is left uninitialized.

      \param init whether to initialize the matrix to zero or leave it uninitialized
     */
    explicit Matrix(Initialization init = Initialization::zero);

    template <typename T>
    explicit Matrix(T) = delete;

    /// Constructor
    /*!
      Constructs a new Matrix from data \e d. If
      \e view==false (the default) the data is copied, otherwise a view to
      it is constructed.

      \param d
        pointer to data
      \param view
        whether the data is to be viewed or copied
     */
    explicit Matrix(ValueType* d, bool view = false);

    /// Constructor
    /*!
      Constructs a new Matrix from data \e d. If
      \e view==false (the default) the data is copied, otherwise a view to
      it is constructed.
      \note a view is currently not possible, the data will be copied!!! a.ger 17.11.2008

      \param d
        pointer to data
      \param view
        whether the data is to be viewed or copied
     */
    explicit Matrix(const ValueType* d, bool view = false);

    /// Constructor
    /*!
      Constructs a new Matrix from data \e d. If
      \e view==false (the default) the data is copied, otherwise a view to
      it is constructed.

      \param d
        matrix to be copied or viewed
      \param view
        whether the data is to be viewed or copied
     */
    explicit Matrix(Core::LinAlg::SerialDenseMatrix& d, bool view = false);

    /// Constructor
    /*!
      Constructs a new Matrix from data \e d. The data is copied.

      \param d
        matrix to be copied
     */
    explicit Matrix(const Core::LinAlg::SerialDenseMatrix& d);

    /**
     * Copy or view data from SerialDenseVector. A dimension mismatch leads to a runtime error.
     */
    explicit Matrix(Core::LinAlg::SerialDenseVector& d, bool view = false);

    /**
     * Copy data from SerialDenseVector to Matrix. A dimension mismatch leads to a runtime error.
     */
    explicit Matrix(const Core::LinAlg::SerialDenseVector& d);

    /// Constructor
    /*!
      Constructs a new Matrix from \e source. If
      \e view==false the data is copied, otherwise a view to
      it is constructed.

      When both an Epetra and a fixed size version of a matrix is needed
      I recommend constructing an Epetra matrix and having a fixed size
      view onto it. That's because Epetra-Views behave differently than
      normal Epetra matrices in some ways, which can lead to tricky bugs.

      \param source
        matrix to take data from
      \param view
        whether the data is to be viewed or copied
     */
    Matrix(Matrix<rows, cols, ValueType>& source, bool view);

    /// Copy constructor
    /*!
      Constructs a new Matrix from source. Unlike
      the Core::LinAlg::SerialDenseMatrix copy constructor this one *always*
      copies the data, even when \e source is a view.

      \param source
        matrix to copy
     */
    Matrix(const Matrix<rows, cols, ValueType>& source);

    /// Copy constructor
    /*!
      Constructs a new Matrix from \e source. If
      \e view==false the data is copied, otherwise a read-only view to
      it is constructed.

      \param source
        matrix to take data from
      \param view
        whether the data is to be viewed or copied

      \note This constructor sets the readonly_ flag, if \e view==true.
            In this case I recommend to use the copy constructor in combination
            with a const qualifier!
     */
    Matrix(const Matrix<rows, cols, ValueType>& source, bool view);

    /// Deconstructor
    ~Matrix();

    /// Return the value_type* holding the data.
    inline const ValueType* data() const { return data_; }
    /// Return the value_type* holding the data.
    inline const ValueType* values() const { return data_; }
    /// Return the value_type* holding the data.
    inline ValueType* data()
    {
      FOUR_C_ASSERT((not isreadonly_), "No write access to read-only data!");
      return data_;
    }
    /// Return the value_type* holding the data.
    inline ValueType* values()
    {
      FOUR_C_ASSERT((not isreadonly_), "No write access to read-only data!");
      return data_;
    }
    /// Return the number of rows
    static constexpr unsigned int m() { return num_rows(); }
    /// Return the number of columns
    static constexpr unsigned int n() { return num_cols(); }
    /// Return the number of rows
    static constexpr unsigned int num_rows() { return rows; }
    /// Return the number of columns
    static constexpr unsigned int num_cols() { return cols; }
    /// Check whether the matrix is initialized
    /*!
      You cannot test whether the matrix is empty using m() and n(),
      for they will always return the templated size. Instead this
      function can be used, it tests whether the data pointer is not
      nullptr.

      \note To actually get a matrix for which is_initialized() returns
      false you must construct a view to nullptr, because the default
      constructor already allocates memory.
     */
    inline bool is_initialized() const { return data() != nullptr; }

    /// Set view
    /*!
      Set this matrix to be a view to \e data.

      \param data
        memory to be viewed
     */
    void set_view(ValueType* data);

    /// Set view
    /*!
      Set this matrix to be a view to \e source.

      \param source
        matrix to be viewed
     */
    void set_view(Matrix<rows, cols, ValueType>& source);

    /// Set copy
    /*!
      Set this matrix to be a copy of \e data. The difference to update(\e data)
      is that this function will allocate it's own memory when it was a
      view before, Update would copy the data into the view.

      \param data
        memory to copy
     */
    void set_copy(const ValueType* data);

    /// Set copy
    /*!
      Set this matrix to be a copy of source. Only the value_type array
      will be copied, the \e isview_ flag is ignored (this is equivalent to
      set_copy(source.values()). The difference to update(\e source) is that this function will
      allocate it's own memory when it was a view before, Update would copy the data into the view.

      \param source
        matrix to copy from
     */
    void set_copy(const Matrix<rows, cols, ValueType>& source);


    /// Calculate determinant
    /*!
      \return determinant
     */
    inline ValueType determinant() const;

    /// invert in place
    /*!
      invert this matrix in place.

      \return determinant of matrix before inversion
     */
    inline ValueType invert();

    /// invert matrix
    /*!
      invert matrix \e other and store the result in \e this.

      \param other
        matrix to be inverted
      \return determinant of \e other
     */
    inline ValueType invert(const Matrix<rows, cols, ValueType>& other);

    /// Set to zero
    /*!
      Sets every value in this matrix to zero. This is equivalent to
      PutScalar(0.0), but it should be faster.
     */
    inline void clear() { DenseFunctions::clear_matrix<ValueType, rows, cols>(data()); }

    /// Fill with scalar
    /*!
      Sets every value in this matrix to \e scalar.

      \param scalar
        value to fill matrix with
     */
    constexpr inline void put_scalar(const ValueType scalar)
    {
      DenseFunctions::put_scalar<ValueType, rows, cols>(scalar, data());
    }

    /// Dot product
    /*!
      Return the dot product of \e this and \e other.

      \param other
        second factor
      \return dot product
     */
    template <class ValueTypeOut = ValueType, class ValueTypeOther>
    constexpr inline ValueTypeOut dot(const Matrix<rows, cols, ValueTypeOther>& other) const
    {
      return DenseFunctions::dot<ValueTypeOut, rows, cols>(data(), other.values());
    }

    /// Cross product
    /*!
      Return the cross product of \e left and \e right.

      \param left
      \param right
      \return cross product
     */
    template <class ValueTypeLeft, class ValueTypeRight>
    inline void cross_product(const Matrix<rows, cols, ValueTypeLeft>& left,
        const Matrix<rows, cols, ValueTypeRight>& right)
    {
      DenseFunctions::crossproduct<ValueType, rows, cols>(data(), left.values(), right.values());
    }

    /// Compute absolute value
    /*!
      Fill this matrix with the absolute value of the numbers in \e other.

      \param other
        matrix to read values from
     */
    constexpr inline void abs(const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::abs<ValueType, rows, cols>(data(), other.values());
    }

    /// Compute reciprocal value
    /*!
      Fill this matrix with the reciprocal value of the numbers in \e other.

      \param other
        matrix to read values from
     */
    constexpr inline void reciprocal(const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::reciprocal<ValueType, rows, cols>(data(), other.values());
    }

    /// Scale
    /*!
      Scale matrix with \e scalar.

      \param scalar
        scaling factor
     */
    inline void scale(const ValueType scalar)
    {
      DenseFunctions::scale_matrix<ValueType, rows, cols>(scalar, data());
    }

    /// Normalize vector
    /*!
      Normalize this matrix, i.e. scale it to have a norm2 of 1.
      This only works for column vectors, i.e. matrices with one column.

      \note This function will throw an exception if the matrix is not a column vector.
     */
    inline void normalize() { DenseFunctions::normalize<ValueType, rows, cols>(data()); }

    /// Copy: \e this = \e other
    /*!
      Copy \e other to \e this.

      \param other
        matrix to copy
     */
    template <class ValueTypeOther>
    constexpr inline void update(const Matrix<rows, cols, ValueTypeOther>& other)
    {
      DenseFunctions::update<ValueType, rows, cols>(data(), other.values());
    }

    /// Scaled copy: \e this = \e scalarOther * \e other
    /*!
      Copy \e scalarOther * \e other to \e this.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to read from
     */
    constexpr inline void update(
        const ValueType scalarOther, const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::update<ValueType, rows, cols>(data(), scalarOther, other.values());
    }

    /// Add: \e this = \e scalarThis * \e this + \e scalarOther * \e other
    /*!
      Scale by \e scalarThis and add \e scalarOther * \e other.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to add
      \param scalarThis
        scaling factor for \e this
     */
    template <class ValueTypeScalarOther, class ValueTypeOther, class ValueTypeScalarThis>
    constexpr inline void update(const ValueTypeScalarOther scalarOther,
        const Matrix<rows, cols, ValueTypeOther>& other, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::update<ValueType, rows, cols>(
          scalarThis, data(), scalarOther, other.values());
    }

    /// Add: \e this = \e left + \e right
    /*!
      Store \e left + \e right in this matrix.

      \param left
        first matrix to add
      \param right
        second matrix to add
     */
    template <class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void update(const Matrix<rows, cols, ValueTypeLeft>& left,
        const Matrix<rows, cols, ValueTypeRight>& right)
    {
      DenseFunctions::update<ValueType, rows, cols>(data(), left.values(), right.values());
    }

    /// Add: \e this = \e scalarLeft * \e left + \e scalarRight * \e right
    /*!
      Store \e scalarLeft * \e left + \e scalarRight * \e right in \e this.

      \param scalarLeft
        scaling factor for \e left
      \param left
        first matrix to add
      \param scalarRight
        scaling factor for \e right
      \param right
        second matrix to add
     */
    template <class ValueTypeScalarLeft, class ValueTypeLeft, class ValueTypeScalarRight,
        class ValueTypeRight>
    constexpr inline void update(const ValueTypeScalarLeft scalarLeft,
        const Matrix<rows, cols, ValueTypeLeft>& left, const ValueTypeScalarRight scalarRight,
        const Matrix<rows, cols, ValueTypeRight>& right)
    {
      DenseFunctions::update<ValueType, rows, cols>(
          data(), scalarLeft, left.values(), scalarRight, right.values());
    }

    /// Add: \e this = \e scalarThis * \e this + \e scalarLeft * \e left + \e scalarRight * \e right
    /*!
      Scale by \e scalarThis and add \e scalarLeft * \e left + \e scalarRight * \e right.

      \param scalarLeft
        scaling factor for \e left
      \param left
        first matrix to add
      \param scalarRight
        scaling factor for \e right
      \param right
        second matrix to add
      \param scalarThis
        scaling factor for \e this
     */
    template <class ValueTypeScalarLeft, class ValueTypeLeft, class ValueTypeScalarRight,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void update(const ValueTypeScalarLeft scalarLeft,
        const Matrix<rows, cols, ValueTypeLeft>& left, const ValueTypeScalarRight scalarRight,
        const Matrix<rows, cols, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::update<ValueType, rows, cols>(
          scalarThis, data(), scalarLeft, left.values(), scalarRight, right.values());
    }

    /// Transposed copy: \e this = \e other^T
    /*!
      Copy transposed \e other to \e this.

      \param other
        matrix to copy
     */
    template <class ValueTypeOther>
    constexpr inline void update_t(const Matrix<cols, rows, ValueTypeOther>& other)
    {
      DenseFunctions::update_t<ValueType, rows, cols>(data(), other.values());
    }

    /// Scaled transposed copy: \e this = \e scalarOther * \e other^T
    /*!
      Transposed copy \e scalarOther * \e other^T to \e this.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to read from
     */
    template <class ValueTypeOtherScalar, class ValueTypeOther>
    constexpr inline void update_t(
        const ValueTypeOtherScalar scalarOther, const Matrix<cols, rows, ValueTypeOther>& other)
    {
      DenseFunctions::update_t<ValueType, rows, cols>(data(), scalarOther, other.values());
    }

    /// Add: \e this = \e scalarThis * \e this + \e scalarOther * \e other
    /*!
      Scale by \e scalarThis and add \e scalarOther * \e other.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to add
      \param scalarThis
        scaling factor for \e this
     */
    template <class ValueTypeOtherScalar, class ValueTypeOther, class ValueTypeThisScalar>
    constexpr inline void update_t(const ValueTypeOtherScalar scalarOther,
        const Matrix<cols, rows, ValueTypeOther>& other, const ValueTypeThisScalar scalarThis)
    {
      DenseFunctions::update_t<ValueType, rows, cols>(
          scalarThis, data(), scalarOther, other.values());
    }

    /// Multiply element-wise: \e this(m,n) *= \e other(m,n)
    /*!
      Multiply \e this and \e other, storing the result in \e this.

      \param other
        factor
     */
    constexpr inline void elementwise_multiply(const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::elementwise_multiply<ValueType, rows, cols>(data(), other.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e scalar * \e this(m,n)*\e other(m,n)
    /*!
      Multiply \e this and \e other, scale by \e scalar and store the result in \e this.

      \param scalar
        scaling factor for the product
      \param other
        factor
     */
    constexpr inline void elementwise_multiply(
        const ValueType scalar, const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::elementwise_multiply<ValueType, rows, cols>(scalar, data(), other.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right and store the result in \e this.

      \param left
        first factor
      \param right
        second factor
     */
    constexpr inline void elementwise_multiply(
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right)
    {
      DenseFunctions::elementwise_multiply<ValueType, rows, cols>(
          data(), left.values(), right.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e scalarOther*\e left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right, scale by \e scalarOther and store the
      result in \e this.

      \param scalarOther
        scaling factor
      \param left
        first factor
      \param right
        second factor
     */
    constexpr inline void elementwise_multiply(const ValueType scalarOther,
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right)
    {
      DenseFunctions::elementwise_multiply<ValueType, rows, cols>(
          data(), scalarOther, left.values(), right.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e scalarThis*\e this(m,n) + \e scalarOther*\e
    /// left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right, scale by \e scalarOther and add the result to \e this, scaled
      by \e scalarThis.

      \param scalarOther
        scaling factor the product
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c i)x(\c j)
      \param scalarThis
        scaling factor for \e this
     */
    constexpr inline void elementwise_multiply(const ValueType scalarOther,
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right,
        const ValueType scalarThis)
    {
      DenseFunctions::elementwise_multiply<ValueType, rows, cols>(
          scalarThis, data(), scalarOther, left.values(), right.values());
    }

    /// Divide element-wise: \e this(m,n) *= \e other(m,n)
    /*!
      Divide \e this by \e other, storing the result in \e this.

      \param other
        factor
     */
    constexpr inline void elementwise_divide(const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::elementwise_divide<ValueType, rows, cols>(data(), other.values());
    }

    /// Divide element-wise: \e this(m,n) = \e scalar * \e this(m,n)*\e other(m,n)
    /*!
      Divide \e this by \e other, scale by \e scalar and store the result in \e this.

      \param scalar
        scaling factor for the product
      \param other
        factor
     */
    constexpr inline void elementwise_divide(
        const ValueType scalar, const Matrix<rows, cols, ValueType>& other)
    {
      DenseFunctions::elementwise_divide<ValueType, rows, cols>(scalar, data(), other.values());
    }

    /// Divide element-wise: \e this(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Divide \e left by \e right and store the result in \e this.

      \param left
        dividend
      \param right
        divisor
     */
    constexpr inline void elementwise_divide(
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right)
    {
      DenseFunctions::elementwise_divide<ValueType, rows, cols>(
          data(), left.values(), right.values());
    }

    /// Divide element-wise: \e this(m,n) = \e scalarOther*\e left(m,n)*\e right(m,n)
    /*!
      Divide \e left by \e right, scale by \e scalarOther and store the
      result in \e this.

      \param scalarOther
        scaling factor
      \param left
        dividend
      \param right
        divisor
     */
    constexpr inline void elementwise_divide(const ValueType scalarOther,
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right)
    {
      DenseFunctions::elementwise_divide<ValueType, rows, cols>(
          data(), scalarOther, left.values(), right.values());
    }

    /// Divide element-wise: \e this(m,n) = \e scalarThis*\e this(m,n) + \e scalarOther*\e
    /// left(m,n)/\e right(m,n)
    /*!
      Divide \e left by \e right, scale by \e scalarOther and add the result to \e this, scaled by
      \e scalarThis.

      \param scalarOther
        scaling factor the product
      \param left
        dividend, size (\c i)x(\c j)
      \param right
        divisor, size (\c i)x(\c j)
      \param scalarThis
        scaling factor for \e this
     */
    constexpr inline void elementwise_divide(const ValueType scalarOther,
        const Matrix<rows, cols, ValueType>& left, const Matrix<rows, cols, ValueType>& right,
        const ValueType scalarThis)
    {
      DenseFunctions::elementwise_divide<ValueType, rows, cols>(
          scalarThis, data(), scalarOther, left.values(), right.values());
    }


    /// Calculate 1-norm
    /*!
      This is *not* the same as Core::LinAlg::SerialDenseMatrix::NormOne.

      \return 1-norm
     */
    constexpr inline ValueType norm1() const
    {
      return DenseFunctions::norm1<ValueType, rows, cols>(data());
    }

    /// Calculate 2-norm (Euclidean norm)
    /*!
      \return 2-norm
     */
    constexpr inline ValueType norm2() const
    {
      return DenseFunctions::norm2<ValueType, rows, cols>(data());
    }

    /// Calculate inf-norm
    /*!
      This is *not* the same as Core::LinAlg::SerialDenseMatrix::NormInf.

      \return inf-norm
     */
    constexpr inline ValueType norm_inf() const
    {
      return DenseFunctions::norm_inf<ValueType, rows, cols>(data());
    }

    /// Calculate minimum value
    /*!
      \return minimum value
     */
    constexpr inline ValueType min_value() const
    {
      return DenseFunctions::min_value<ValueType, rows, cols>(data());
    }

    /// Calculate maximum value
    /*!
      \return maximum value
     */
    constexpr inline ValueType max_value() const
    {
      return DenseFunctions::max_value<ValueType, rows, cols>(data());
    }

    /// Calculate mean value
    /*!
      \return mean value
     */
    constexpr inline ValueType mean_value() const
    {
      return DenseFunctions::mean_value<ValueType, rows, cols>(data());
    }

    /// Multiply: \e this = \e left*right
    /*!
      This is equivalent to multiply_nn(\e left,\e right).

      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void multiply(const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(data(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left*right
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void multiply_nn(const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(data(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left*right^T
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void multiply_nt(const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_nt<ValueType, rows, inner, cols>(
          data(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left^T*right
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void multiply_tn(const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_tn<ValueType, rows, inner, cols>(
          data(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left^T*right^T
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeLeft, class ValueTypeRight>
    constexpr inline void multiply_tt(const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_tt<ValueType, rows, inner, cols>(
          data(), left.values(), right.values());
    }


    /// Multiply: \e this = \e scalarOthers * \e left*right
    /*!
      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight>
    constexpr inline void multiply(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(
          data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarOthers * \e left*right
    /*!
      This is equivalent to multiply_nn(\e scalarOthers,\e left,\e right).

      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight>
    constexpr inline void multiply_nn(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(
          data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarOthers * \e left*right^T
    /*!
      \param scalarOthers
        scalar factor for \e left*right^T
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight>
    constexpr inline void multiply_nt(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_nt<ValueType, rows, inner, cols>(
          data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarOthers * \e left^T*right
    /*!
      \param scalarOthers
        scalar factor for \e left^T*right
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight>
    constexpr inline void multiply_tn(const ValueTypeScalarOther scalarOthers,
        const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_tn<ValueType, rows, inner, cols>(
          data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarOthers * \e left^T*right^T
    /*!
      \param scalarOthers
        scalar factor for \e left^T*right^T
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight>
    constexpr inline void multiply_tt(const ValueTypeScalarOther scalarOthers,
        const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right)
    {
      DenseFunctions::multiply_tt<ValueType, rows, inner, cols>(
          data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left*right
    /*!
      This is equivalent to multiply_nn(\e scalarOthers,\e left,\e right,\e scalarThis).

      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void multiply(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(
          scalarThis, data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left*right
    /*!
      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void multiply_nn(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::multiply<ValueType, rows, inner, cols>(
          scalarThis, data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left*right^T
    /*!
      \param scalarOthers
        scalar factor for \e left*right^T
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void multiply_nt(const ValueTypeScalarOther scalarOthers,
        const Matrix<rows, inner, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::multiply_nt<ValueType, rows, inner, cols>(
          scalarThis, data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left^T*right
    /*!
      \param scalarOthers
        scalar factor for \e left^T*right
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void multiply_tn(const ValueTypeScalarOther scalarOthers,
        const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<inner, cols, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::multiply_tn<ValueType, rows, inner, cols>(
          scalarThis, data(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left^T*right^T
    /*!
      \param scalarOthers
        scalar factor for \e left^T*right^T
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class ValueTypeScalarOther, class ValueTypeLeft,
        class ValueTypeRight, class ValueTypeScalarThis>
    constexpr inline void multiply_tt(const ValueTypeScalarOther scalarOthers,
        const Matrix<inner, rows, ValueTypeLeft>& left,
        const Matrix<cols, inner, ValueTypeRight>& right, const ValueTypeScalarThis scalarThis)
    {
      DenseFunctions::multiply_tt<ValueType, rows, inner, cols>(
          scalarThis, data(), scalarOthers, left.values(), right.values());
    }

    /// Write \e this to \e out
    /*!
      Write a readable representation of \e this to \e out. This function is called by
      \e out << *\e this.

      \param out
        out stream
     */
    void print(std::ostream& out) const;

    /// = operator
    /*!
      Copy data from \e other to \e this, equivalent to update(other).
      \param other
        matrix to get data from
     */
    inline Matrix<rows, cols, ValueType>& operator=(const Matrix<rows, cols, ValueType>& other);

    /// = operator for double
    /*!
      Fill with double \e other, same as PutScalar(other).

      \param other
        scalar value
     */
    inline Matrix<rows, cols, ValueType>& operator=(const ValueType other);

    /// == operator
    /*!
      Compare \e this with \e other.

      \param other
        matrix to compare with
     */
    inline bool operator==(const Matrix<rows, cols, ValueType>& other) const;

    /// != operator
    /*!
      Compare \e this with \e other.

      \param other
        matrix to compare with
     */
    inline bool operator!=(const Matrix<rows, cols, ValueType>& other) const;

    /// += operator
    /*!
      Add \e other to \e this.

      \param other
        matrix to add
     */
    template <class ValueTypeOther>
    inline Matrix<rows, cols, ValueType>& operator+=(
        const Matrix<rows, cols, ValueTypeOther>& other)
    {
      DenseFunctions::update<ValueType, rows, cols>(1.0, data(), 1.0, other.values());
      return *this;
    }

    /// -= operator
    /*!
      Subtract \e other from \e this.

      \param other
        matrix to subtract
     */
    template <class ValueTypeOther>
    inline Matrix<rows, cols, ValueType>& operator-=(
        const Matrix<rows, cols, ValueTypeOther>& other)
    {
      DenseFunctions::update<ValueType, rows, cols>(1.0, data(), -1.0, other.values());
      return *this;
    }

    /// Access data
    /*!
      Return value in row \e r and column \e c.

      \param r
        row index
      \param c
        column index
     */
    inline ValueType& operator()(unsigned int r, unsigned int c);

    /// Access data
    /*!
      Return value in row \e r and column \e c.

      \param r
        row index
      \param c
        column index
     */
    inline const ValueType& operator()(unsigned int r, unsigned int c) const;

    /// Access data
    /*!
      Return value in row \e r. This works only for Matrices with cols==1 or rows==1 (vectors),
      otherwise a compile time error is raised.

      \param r
        index
     */
    inline ValueType& operator()(unsigned int r);  // for vectors, with check at compile-time

    /// Access data
    /*!
      Return value in row \e r. This works only for Matrices with cols==1 or rows==1 (vectors),
      otherwise a compile time error is raised.

      \param r
        index
     */
    inline const ValueType& operator()(unsigned int r) const;
  };

  template <class ValueType, unsigned int cols, unsigned int rows>
  std::ostream& operator<<(std::ostream& out, const Matrix<rows, cols, ValueType>& matrix);

  // Constructors

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(Initialization init)
      : data_(nullptr), isview_(false), isreadonly_(false)
  {
    if (allocatesmemory_)
      data_ = new ValueType[rows * cols];
    else
      data_ = datafieldsmall_;
    if (init == Initialization::zero) DenseFunctions::clear_matrix<ValueType, rows, cols>(data_);
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(ValueType* d, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    if (isview_)
    {
      data_ = d;
    }
    else
    {
      if (allocatesmemory_)
        data_ = new ValueType[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(d, d + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(const ValueType* d, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    if (isview_)
    {
      isreadonly_ = true;
      data_ = const_cast<ValueType*>(d);
    }
    else
    {
      if (allocatesmemory_)
        data_ = new ValueType[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(d, d + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(Core::LinAlg::SerialDenseMatrix& d, bool view)
      : Matrix(Internal::values(d), view)
  {
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(const Core::LinAlg::SerialDenseMatrix& d)
      : Matrix(Internal::values(d), false)
  {
  }


  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(Core::LinAlg::SerialDenseVector& d, bool view)
      : Matrix(Internal::values(d), view)
  {
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(const Core::LinAlg::SerialDenseVector& d)
      : Matrix(Internal::values(d), false)
  {
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(Matrix<rows, cols, ValueType>& source, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    if (isview_)
    {
      data_ = source.data_;
    }
    else
    {
      if (allocatesmemory_)
        data_ = new ValueType[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(source.data_, source.data_ + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(const Matrix<rows, cols, ValueType>& source)
      : data_(nullptr), isview_(false), isreadonly_(false)
  {
    if (allocatesmemory_)
      data_ = new ValueType[rows * cols];
    else
      data_ = datafieldsmall_;
    std::copy(source.data_, source.data_ + rows * cols, data_);
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::Matrix(const Matrix<rows, cols, ValueType>& source, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    if (isview_)
    {
      isreadonly_ = true;
      data_ = const_cast<ValueType*>(source.values());
    }
    else
    {
      if (allocatesmemory_)
        data_ = new ValueType[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(source.data_, source.data_ + rows * cols, data_);
    }
  }

  // Destructor
  template <unsigned int rows, unsigned int cols, class ValueType>
  Matrix<rows, cols, ValueType>::~Matrix()
  {
    if (allocatesmemory_ && not isview_) delete[] data_;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  void Matrix<rows, cols, ValueType>::set_view(ValueType* data)
  {
    if (not isview_)
    {
      if (allocatesmemory_) delete[] data_;
      isview_ = true;
    }
    data_ = data;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  void Matrix<rows, cols, ValueType>::set_view(Matrix<rows, cols, ValueType>& source)
  {
    if (not isview_)
    {
      if (allocatesmemory_) delete[] data_;
      isview_ = true;
    }
    data_ = source.data_;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  void Matrix<rows, cols, ValueType>::set_copy(const ValueType* data)
  {
    if (isview_)
    {
      if (allocatesmemory_)
        data_ = new ValueType[rows * cols];
      else
        data_ = datafieldsmall_;
      isview_ = false;
    }
    std::copy(data, data + rows * cols, data_);
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  void Matrix<rows, cols, ValueType>::set_copy(const Matrix<rows, cols, ValueType>& source)
  {
    set_copy(source.values());
  }

  // Determinant
  template <unsigned int rows, unsigned int cols, class ValueType>
  inline ValueType Matrix<rows, cols, ValueType>::determinant() const
  {
    static_assert(rows == cols, "Cannot compute determinant of non-square matrix.");
    return DenseFunctions::determinant<ValueType, rows, cols>(data());
  }

  // invert
  template <unsigned int rows, unsigned int cols, class ValueType>
  inline ValueType Matrix<rows, cols, ValueType>::invert()
  {
    static_assert(rows == cols, "Cannot compute inverse of non-square matrix");
    return DenseFunctions::invert<ValueType, rows, cols>(data());
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline ValueType Matrix<rows, cols, ValueType>::invert(const Matrix<rows, cols, ValueType>& other)
  {
    static_assert(rows == cols, "Cannot compute inverse of non-square matrix");
    return DenseFunctions::invert<ValueType, rows, cols>(data(), other.values());
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  void Matrix<rows, cols, ValueType>::print(std::ostream& out) const
  {
    out << "Matrix<" << rows << ',' << cols << '>';
    if (isview_) out << " (view to memory only)";
    if (isreadonly_) out << "\n (read only)";
    if (data() == nullptr)
    {
      out << " with data_==nullptr!\n";
      return;
    }
    if (cols > 1)
    {
      out << "\n[";
      for (unsigned int i = 0; i < rows; ++i)
      {
        if (i != 0) out << ' ';
        for (unsigned int j = 0; j < cols; ++j)
        {
          out << data()[i + rows * j];
          if (j + 1 < cols) out << ", ";
        }
        if (i + 1 < rows)
          out << ",\n";
        else
          out << "]\n";
      }
    }
    else
    {
      out << "[";
      for (unsigned int i = 0; i < rows; ++i)
      {
        if (i != 0) out << ' ';
        out << data()[i];
      }
      out << "]\n";
    }
  }

  /// output operator for Matrix
  /*!
    Write matrix to out. This function calls matrix.print(out).
   */
  template <class ValueType, unsigned int cols, unsigned int rows>
  std::ostream& operator<<(std::ostream& out, const Matrix<rows, cols, ValueType>& matrix)
  {
    matrix.print(out);
    return out;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline Matrix<rows, cols, ValueType>& Matrix<rows, cols, ValueType>::operator=(
      const Matrix<rows, cols, ValueType>& other)
  {
    DenseFunctions::update<ValueType, rows, cols>(data(), other.values());
    return *this;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline Matrix<rows, cols, ValueType>& Matrix<rows, cols, ValueType>::operator=(
      const ValueType other)
  {
    DenseFunctions::put_scalar<ValueType, rows, cols>(other, data());
    return *this;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline bool Matrix<rows, cols, ValueType>::operator==(
      const Matrix<rows, cols, ValueType>& other) const
  {
    if (data() == other.values()) return true;
    // unfortunately memcmp does not work, because +0 and -0 are
    // different in memory...
    for (unsigned int c = 0; c < rows * cols; ++c)
      if (data()[c] != other.values()[c]) return false;
    return true;
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline bool Matrix<rows, cols, ValueType>::operator!=(
      const Matrix<rows, cols, ValueType>& other) const
  {
    return not(*this == other);
  }

  // Access operator

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline ValueType& Matrix<rows, cols, ValueType>::operator()(unsigned int r, unsigned int c)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= rows or c >= cols)
      FOUR_C_THROW("Indices {},{} out of range in Matrix<{},{}>.", r, c, rows, cols);
#endif
    return data()[r + c * rows];
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline const ValueType& Matrix<rows, cols, ValueType>::operator()(
      unsigned int r, unsigned int c) const
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= rows or c >= cols)
      FOUR_C_THROW("Indices {},{} out of range in Matrix<{},{}>.", r, c, rows, cols);
#endif
    return data()[r + c * rows];
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline ValueType& Matrix<rows, cols, ValueType>::operator()(unsigned int r)
  {
    static_assert((cols == 1) or (rows == 1), "cannot call 1-d access function on 2-d matrix");
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= (cols == 1 ? rows : cols))
      FOUR_C_THROW("Index {} out of range in Matrix<{},{}>.", r, rows, cols);
#endif
    return data()[r];
  }

  template <unsigned int rows, unsigned int cols, class ValueType>
  inline const ValueType& Matrix<rows, cols, ValueType>::operator()(unsigned int r) const
  {
    static_assert((cols == 1) or (rows == 1), "cannot call 1-d access function on 2-d matrix");
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= (cols == 1 ? rows : cols))
      FOUR_C_THROW("Index {} out of range in Matrix<{},{}>.", r, rows, cols);
#endif
    return data()[r];
  }
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file

\brief a templated fixed size dense matrix

\level 0
*/
/*----------------------------------------------------------------------*/
#ifndef FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP
#define FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP

#include "baci_config.hpp"

#include "baci_linalg_serialdensematrix.hpp"
#include "baci_utils_exceptions.hpp"
#include "baci_utils_mathoperations.hpp"

#include <Teuchos_BLAS.hpp>
#include <Teuchos_LAPACK.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <ostream>
#include <vector>

FOUR_C_NAMESPACE_OPEN

// Attention: In the case a CORE::LINALG::Matrix is created with float_type ClnWrapper,
// include the header "/cut/cut_clnwrapper.H" before this header is processed.

namespace CORE::LINALG
{
  namespace DENSEERROR
  {
    /// Compile time error definitions
    /*!
      A struct that's used for compile time error checking. The
      template argument is the expression to be checked, if it
      evaluates to true nothing happens. If it is false a compile
      error similiar to
      "'Matrix_dimensions_cannot_be_zero' is not a member of
      'CORE::LINALG::DENSEERROR::Checker<false>'" is generated (with gcc). Obviously the test
      expression must be known at compile time.

      This is the (general) definition that is used for expr==true. It
      defines all known errors als empty static inline functions, so
      that the compiler can optimize them away.
     */
    template <bool expr>
    struct Checker
    {
      static inline void Matrix_dimensions_cannot_be_zero(){};
      static inline void Cannot_call_1D_access_function_on_2D_matrix(){};
      static inline void Cannot_compute_determinant_of_nonsquare_matrix(){};
      static inline void Cannot_compute_inverse_of_nonsquare_matrix(){};
      static inline void Transpose_argument_must_be_N_or_T(){};
      static inline void Matrix_size_in_solver_must_be_square(){};
      static inline void Use_FixedSizeSerialDenseSolver_for_matrices_bigger_than_3x3(){};
    };

    /// Compile time error definitions: missing functions raise errors
    /*!
      This is the specialisation for expr==false. It is empty, so that
      the compiler does not find the functions and raises errors.
     */
    template <>
    struct Checker<false>
    {
    };

  }  // namespace DENSEERROR

  namespace DENSEFUNCTIONS
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
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiply(value_type_out* out, const value_type_left* const left,
        const value_type_right* const right);

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
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyNN(value_type_out* out, const value_type_left* const left,
        const value_type_right* const right);

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
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyNT(value_type_out* out, const value_type_left* const left,
        const value_type_right* const right);

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
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyTN(value_type_out* out, const value_type_left* const left,
        const value_type_right* const right);

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
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyTT(value_type_out* out, const value_type_left* const left,
        const value_type_right* const right);

    /// Multiplication: \e out = \e infac * \e left*\e right
    /*!
      Multiply \e left and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template parameters \c
      i, \c j and \c k denoting the sizes of the matrices.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiply(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right);

    /// Multiplication: \e out = \e infac * \e left*\e right
    /*!
      Multiply \e left and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template parameters \c
      i, \c j and \c k denoting the sizes of the matrices.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyNN(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right);

    /// Multiplication: \e out = \e infac * \e left*\e right^T
    /*!
      Multiply \e left and \e right^T, scale the result by \e infac and store
      it in \e out. This function takes three template parameters \c
      i, \c j and \c k denoting the sizes of the matrices.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right^T
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c k)x(\c j) so that \e
        right^T has size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyNT(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right);

    /// Multiplication: \e out = \e infac * \e left^T*\e right
    /*!
      Multiply \e left^T and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template parameters \c
      i, \c j and \c k denoting the sizes of the matrices.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left^T*right
      \param left
        pointer to the first factor, size (\c j)x(\c i) so that \e
        left^T has size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyTN(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right);

    /// Multiplication: \e out = \e infac * \e left^T*\e right^T
    /*!
      Multiply \e left^T and \e right^T, scale the result by \e infac and store
      it in \e out. This function takes three template parameters \c
      i, \c j and \c k denoting the sizes of the matrices.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left^T*right^T
      \param left
        pointer to the first factor, size (\c j)x(\c i) so that \e
        left^T has size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c k)x(\c j) so that \e
        right^T has size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyTT(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right);

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
        scalar to muliply with \e left*right
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiply(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right);

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
        scalar to muliply with \e left*right
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyNN(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right);

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
        scalar to muliply with \e left*right^T
      \param left
        pointer to the first factor, size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c k)x(\c j) so that \e
        right^T has size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyNT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right);

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
        scalar to muliply with \e left^T*right
      \param left
        pointer to the first factor, size (\c j)x(\c i) so that \e
        left^T has size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyTN(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right);

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
        scalar to muliply with \e left^T*right^T
      \param left
        pointer to the first factor, size (\c j)x(\c i) so that \e
        left^T has size (\c i)x(\c j)
      \param right
        pointer to the second factor, size (\c k)x(\c j) so that \e
        right^T has size (\c j)x(\c k)
     */
    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyTT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right);

    /// Invert matrix: \e out = inv(\e in)
    /*!
      Invert the matrix \e in and store the result in \e out. To keep a
      common interface there are two template parameters \c i and \c j, but
      they must be the same number. The sizes of \e in and \e out are
      expected to be (\c i)x(\c j), and they must be square.

      \param out
        pointer to the memory the result should be stored in, size (\c i)x(\c j)
      \param in
        pointer to the matrix to be inverted, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(value_type* out, const value_type* in);

    /// Invert matrix: \e mat = inv(\e mat)
    /*!
      Invert the matrix \e mat in place. To keep a common interface there
      are two template parameters \c i and \c j, but they must be the same
      number. The size of \e mat is expected to be (\c i)x(\c j), and it must
      be square.

      \param mat
        pointer to the matrix to be inverted in place, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(value_type* mat);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type determinant(const value_type* mat);

    /// Copy: \e out = \e in
    /*!
      Copy \e in to \e out. This function takes two template parameters \c i and
      \c j denoting the sizes of the matrices.

      \param out
        pointer to the result matrix, size (\c i)x(\c j)
      \param in
        pointer to the matrix to be copied, size (\c i)x(\c j)
     */
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_in>
    inline void update(value_type_out* out, const value_type_in* in);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_infac,
        class value_type_in>
    inline void update(value_type_out* out, const value_type_infac infac, const value_type_in* in);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_infac, class value_type_in>
    inline void update(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_in* in);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_left,
        class value_type_right>
    inline void update(
        value_type_out* out, const value_type_left* left, const value_type_right* right);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_leftfac,
        class value_type_left, class value_type_rightfac, class value_type_right>
    inline void update(value_type_out* out, const value_type_leftfac leftfac,
        const value_type_left* left, const value_type_rightfac rightfac,
        const value_type_right* right);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_leftfac, class value_type_left, class value_type_rightfac,
        class value_type_right>
    inline void update(const value_type_outfac outfac, value_type_out* out,
        const value_type_leftfac leftfac, const value_type_left* left,
        const value_type_rightfac rightfac, const value_type_right* right);

    /// Transposed copy: \e out = \e in^T
    /*!
      Copy transposed \e in to \e out. This function takes two template parameters \c i and
      \c j denoting the sizes of the matrices.

      \param out
        pointer to the result matrix, size (\c i)x(\c j)
      \param in
        pointer to the matrix to be copied, size (\c j)x(\c i)
     */
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_in>
    inline void updateT(value_type_out* out, const value_type_in* in);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_infac,
        class value_type_in>
    inline void updateT(value_type_out* out, const value_type_infac infac, const value_type_in* in);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_infac, class value_type_in>
    inline void updateT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_in* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(value_type* out, const value_type* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type fac, value_type* out, const value_type* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(value_type* out, const value_type* left, const value_type* right);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(
        value_type* out, const value_type infac, const value_type* left, const value_type* right);

    /// Multiply element-wise, \e out(m,n) = \e outfac*\e out(m,n) + \e infac*\e left(m,n)*\e
    /// right(m,n)
    /*!
      Multiply \e left and \e right, scale by \e infac and add the result to \e out, scaled by \e
      outfac. This function takes two template parameters, unsigned ints \c i and \c j denoting the
      size of the matrices. \param outfac scaling factor for \e out \param out pointer to result,
      size (\c i)x(\c j) \param infac scaling factor the product \param left pointer to first
      factor, size (\c i)x(\c j) \param right pointer to second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type outfac, value_type* out, const value_type infac,
        const value_type* left, const value_type* right);

    /// Divide element-wise, \e out(m,n) = \e out(m,n)/\e in(m,n)
    /*!
      Devide \e out by \e in, storing the result in \e out.
      This function takes two template parameters, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        pointer to dividend and result, size (\c i)x(\c j)
      \param in
        pointer to divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(value_type* out, const value_type* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type fac, value_type* out, const value_type* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(value_type* out, const value_type* left, const value_type* right);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(
        value_type* out, const value_type infac, const value_type* left, const value_type* right);

    /// Divide element-wise, \e out(m,n) = \e outfac*\e out(m,n) + \e infac*\e left(m,n)/\e
    /// right(m,n)
    /*!
      Divide \e left by \e right, scale by \e infac and add the result to \e out, scaled by \e
      outfac. This function takes two template parameters, unsigned ints \c i and \c j denoting the
      size of the matrices. \param outfac scaling factor for \e out \param out pointer to result,
      size (\c i)x(\c j) \param infac scaling factor the product \param left pointer to dividend,
      size (\c i)x(\c j) \param right pointer to divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type outfac, value_type* out, const value_type infac,
        const value_type* left, const value_type* right);

    /// Scale matrix
    /*!
      Scale \e mat by \e fac. This function takes
      two template parameters \c i and \c j denoting the size of \e mat.

      \param fac
        scalar to multiply with \e mat
      \param mat
        pointer to the matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void scaleMatrix(const value_type factor, value_type* mat);

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
    template <class value_type_out, unsigned int i, unsigned int j, class value_type_left,
        class value_type_right>
    inline value_type_out dot(const value_type_left* left, const value_type_right* right);

    /// Set matrix to zero
    /*!
      Set matrix \e mat to zero. This function takes two template
      parameters i and j denoting the size of the matrix.

      This is the same as \e putScalar<\c i, \c j>(0.0, \e mat), but it should be faster.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void clearMatrix(value_type* mat);

    /// Fill matrix with scalar value
    /*!
      Set every number in \e mat to \e scalar. This function takes two template
      parameters \c i and \c j denoting the size of the matrix.

      \param scalar
        scalar value to be set
      \param mat
        pointer to the matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void putScalar(const value_type scalar, value_type* mat);

    /// Calculate absolut values of a matrix
    /*!
      Fill \e out with the absolute values from \e in. This function takes two
      template parameters \c i and \c j denoting the sizes of the matrices.

      \param out
        pointer to the matrix to be set, size (\c i)x(\c j)
      \param in
        pointer to the matrix the values are read from, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void abs(value_type* out, const value_type* in);

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
    template <class value_type, unsigned int i, unsigned int j>
    inline void reciprocal(value_type* out, const value_type* in);

    /// 1-norm
    /*!
      This function computes the norm of the whole matrix. It returns
      a different result than CORE::LINALG::SerialDenseMatrix::Base::OneNorm(),
      which returns the maximum of the norms of the columns.
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return 1-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm1(const value_type* mat);

    /// 2-norm (Euclidean norm)
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return 2-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm2(const value_type* mat);

    /// Inf-norm
    /*!
      This function does not do the same as CORE::LINALG::SerialDenseMatrix::Base::InfNorm().
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return inf-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type normInf(const value_type* mat);

    /// Minimum value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return minimum value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type minValue(const value_type* mat);

    /// Maximum value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return maximum value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type maxValue(const value_type* mat);

    /// Mean value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        pointer to the matrix, size (\c i)x(\c j)
      \return mean value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type meanValue(const value_type* mat);

    /*
     * Declaration of the functions taking CORE::LINALG::SerialDenseMatrix::Base
     *
     */


    /// Multiplication: \e out = \e left*\e right
    /*!
      Multiply \e left and \e right and store the result in \e out. This
      function takes three template parameters \c i, \c j and \c
      k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e left*\e right
    /*!
      Multiply \e left and \e right and store the result in \e out. This
      function takes three template parameters \c i, \c j and \c
      k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e left*\e right^T
    /*!
      Multiply \e left and \e right^T and store the result in \e out. This
      function takes three template parameters \c i, \c j and \c
      k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size(\c
        j)x(\e k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e left^T*\e right
    /*!
      Multiply \e left^T and \e right and store the result in \e out. This
      function takes three template parameters \c i, \c j and \c
      k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size(\c
        i)x(\e j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e left^T*\e right^T
    /*!
      Multiply \e left^T and \e right^T and store the result in \e out. This
      function takes three template parameters \c i, \c j and \c
      k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size(\c
        i)x(\e j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size(\c
        j)x(\e k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e infac * \e left*\e right
    /*!
      Multiply \e left and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template
      parameters \c i, \c j and \c k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e infac * \e left*\e right
    /*!
      Multiply \e left and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template
      parameters \c i, \c j and \c k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e infac * \e left*\e right^T
    /*!
      Multiply \e left and \e right^T, scale the result by \e infac and store
      it in \e out. This function takes three template
      parameters \c i, \c j and \c k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e infac * \e left^T*\e right
    /*!
      Multiply \e left^T and \e right, scale the result by \e infac and store
      it in \e out. This function takes three template
      parameters \c i, \c j and \c k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size(\c
        i)x(\e j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e infac * \e left^T*\e right^T
    /*!
      Multiply \e left^T and \e right^T, scale the result by \e infac and store
      it in \e out. This function takes three template
      parameters \c i, \c j and \c k denoting the sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size(\c
        i)x(\e j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right
    /*!
      Scale \e out by \e outfac and add \e left*\e right scaled by \e
      infac. This function takes three template parameters \c i, \c j
      and \c k denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right
    /*!
      Scale \e out by \e outfac and add \e left*\e right scaled by \e
      infac. This function takes three template parameters \c i, \c j
      and \c k denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left*\e right^T
    /*!
      Scale \e out by \e outfac and add \e left*\e right^T scaled by \e
      infac. This function takes three template parameters \c i, \c j
      and \c k denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size
        (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left^T*\e right
    /*!
      Scale \e out by \e outfac and add \e left^T*\e right scaled by \e
      infac. This function takes three template parameters \c i, \c j
      and \c k denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size
        (\c i)x(\c j)
      \param right
        second factor, size (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiplication: \e out = \e outfac * \e out + \e infac * \e left^T*\e right^T
    /*!
      Scale \e out by \e outfac and add \e left^T*\e right^T scaled by \e
      infac. This function takes three template parameters \c i, \c j
      and \c k denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        matrix the result should be stored in, size (\c i)x(\c k)
      \param infac
        scalar to muliply with \e left*right
      \param left
        first factor, size (\c j)x(\c i) so that \e left^T has size
        (\c i)x(\c j)
      \param right
        second factor, size (\c k)x(\c j) so that \e right^T has size
        (\c j)x(\c k)
     */
    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Invert matrix: \e out = inv(\e in)
    /*!
      Invert the matrix \e in and store the result in \e out. To keep a
      common interface there are two template parameters \c i and \c j, but
      they must be the same number. The sizes of \e in and \e out are
      expected to be (\c i)x(\c j), and they must be square.

      \note This function only works for matrices with sizes up to
      3x3. For larger matrices use the FixedSizeSerialDenseSolver.

      \param out
        matrix the result should be stored in, size (\c i)x(\c j)
      \param in
        matrix to be inverted, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& in);

    /// Invert matrix: \e mat = inv(\e mat)
    /*!
      Invert the matrix \e mat in place. To keep a common interface there
      are two template parameters \c i and \c j, but they must be the same
      number. The size of \e mat is expected to be (\c i)x(\c j), and it must
      be square.

      \note This function only works for matrices with sizes up to
      3x3. For larger matrices use the FixedSizeSerialDenseSolver.

      \param mat
        matrix to be inverted in place, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(CORE::LINALG::SerialDenseMatrix::Base& mat);

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

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type determinant(const CORE::LINALG::SerialDenseMatrix::Base& mat);

    /// Copy: \e out = \e in
    /*!
      Copy \e in to \e out. This function takes two template parameters \c i and
      \c j denoting the sizes of the matrices.

      \param out
        result matrix, size (\c i)x(\c j)
      \param in
        matrix to be copied, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& in);

    /// Scaled copy: \e out = \e infac * \e in
    /*!
      Scale \e in by \e infac and store the result in \e out. This function takes two template
      parameters \c i and \c j denoting the sizes of the matrices.

      \param out
        result matrix, size (\c i)x(\c j)
      \param infac
        scalar to multiply with \e in
      \param in
        matrix to read from, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& in);

    /// Addition: \e out = \e outfac * \e out + \e infac * \e in
    /*!
      Scale \e out by \e outfac and add \e infac * \e in to it. This function
      takes two template parameters \c i and \c j denoting the sizes of the matrices.

      \param outfac
        scalar to multiply with \e out
      \param out
        result matrix, size (\c i)x(\c j)
      \param infac
        scalar to multiply with \e in
      \param in
        matrix to be added, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& in);

    /// Addition: \e out = \e left + \e right
    /*!
      Add \e left and \e right and store the result in \e out. This
      function takes two template parameters \c i and \c j denoting the
      sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c j)
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Addition: \e out = \e leftfac * \e left + \e rightfac * \e right
    /*!
      Add \e left and \e right, scaled by \e leftfac and \e rightfac
      respectively. The result is stored in \e out. This
      function takes two template parameters \c i and \c j denoting the
      sizes of the matrices.

      \param out
        matrix the result should be stored in, size (\c i)x(\c j)
      \param leftfac
        scalar to multiply with \e left
      \param left
        first factor, size (\c i)x(\c j)
      \param rightfac
        scalar to multiply with \e right
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type leftfac,
        const CORE::LINALG::SerialDenseMatrix::Base& left, const value_type rightfac,
        const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Addition: \e out = \e outfac * \e out + \e leftfac * \e left + \e rightfac * \e right
    /*!
      Scale \e out by \e outfac and add \e left and \e right, scaled by \e leftfac and \e rightfac
      respectively. The result is stored in \e out. This
      function takes two template parameters \c i and \c j denoting the
      sizes of the matrices.

      \param outfac
        scalar to multiply \e out with
      \param out
        matrix the result should be stored in, size (\c i)x(\c j)
      \param leftfac
        scalar to multiply with \e left
      \param left
        first factor, size (\c i)x(\c j)
      \param rightfac
        scalar to multiply with \e right
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void update(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type leftfac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const value_type rightfac, const CORE::LINALG::SerialDenseMatrix::Base& right);

    /// Multiply element-wise, \e out(m,n) = \e out(m,n)*\e in(m,n)
    /*!
      Multiply \e out and \e in, storing the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        first factor and result, size (\c i)x(\c j)
      \param in
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(
        CORE::LINALG::SerialDenseMatrix::Base out, const CORE::LINALG::SerialDenseMatrix::Base in);

    /// Multiply element-wise, \e out(m,n) = \e fac*\e out(m,n)*\e in(m,n)
    /*!
      Multiply \e out and \e in, scale by \e fac and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param fac
        scaling factor for the product
      \param out
        first factor and result, size (\c i)x(\c j)
      \param in
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type fac, CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base in);

    /// Multiply element-wise, \e out(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        result, size (\c i)x(\c j)
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Multiply element-wise, \e out(m,n) = \e infac*\e left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right, scale by \e infac and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        result, size (\c i)x(\c j)
      \param infac
        scaling factor
      \param left
        first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(CORE::LINALG::SerialDenseMatrix::Base out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Multiply element-wise, \e out(m,n) = \e outfac*out(m,n) + \e infac*\e left(m,n)*\e
    /// right(m,n)
    /*!
      Multiply \e left and \e right, scale by \e infac and add the result to \e out, scaled by \e
      outfac. This function takes two template argumens, unsigned ints \c i and \c j denoting the
      size of the matrices. \param outfac scaling factor for \e out \param out result, size (\c
      i)x(\c j) \param infac scaling factor the product \param left first factor, size (\c i)x(\c j)
      \param right
        second factor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Multiply element-wise, \e out(m,n) = \e out(m,n)*\e in(m,n)
    /*!
      Multiply \e out and \e in, storing the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        dividend and result, size (\c i)x(\c j)
      \param in
        divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(
        CORE::LINALG::SerialDenseMatrix::Base out, const CORE::LINALG::SerialDenseMatrix::Base in);

    /// Divide element-wise, \e out(m,n) = \e fac*\e out(m,n)*\e in(m,n)
    /*!
      Divide \e out and \e in, scale by \e fac and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param fac
        scaling factor for the product
      \param out
        dividend and result, size (\c i)x(\c j)
      \param in
        divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type fac, CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base in);

    /// Divide element-wise, \e out(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Divide \e left and \e right and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        result, size (\c i)x(\c j)
      \param left
        dividend, size (\c i)x(\c j)
      \param right
        divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Divide element-wise, \e out(m,n) = \e infac*\e left(m,n)*\e right(m,n)
    /*!
      Divide \e left and \e right, scale by \e infac and store the result in \e out.
      This function takes two template argumens, unsigned ints \c i and \c j
      denoting the size of the matrices.
      \param out
        result, size (\c i)x(\c j)
      \param infac
        scaling factor
      \param left
        dividend, size (\c i)x(\c j)
      \param right
        divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(CORE::LINALG::SerialDenseMatrix::Base out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Divide element-wise, \e out(m,n) = \e outfac*\e out(m,n) + \e infac*\e left(m,n)*\e
    /// right(m,n)
    /*!
      Divide \e left by \e right, scale by \e infac and add the result to \e out, scaled by \e
      outfac. This function takes two template argumens, unsigned ints \c i and \c j denoting the
      size of the matrices. \param outfac scaling factor for \e out \param out result, size (\c
      i)x(\c j) \param infac scaling factor the product \param left dividend, size (\c i)x(\c j)
      \param right
        divisor, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right);

    /// Scale matrix
    /*!
      Scale \e mat by \e scalar. This function takes
      two template parameters \c i and \c j denoting the size of \e mat.

      \param scalar
        scalar to multiply with \e mat
      \param mat
        matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void scaleMatrix(const value_type scalar, CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      mat.scale(scalar);
    }

    /// Dot product
    /*!
      Return dot product \e left and \e right. This function
      takes two template parameters \c i and \c j denoting the sizes of the matrices.

      \param left
        first matrix, size (\c i)x(\c j)
      \param right
        second matrix, size (\c i)x(\c j)
      \return dot product
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type dot(const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
      return dot<value_type, i, j>(left.values(), right.values());
    }

    /// Set matrix to zero
    /*!
      Set matrix \e mat to zero. This function takes two template
      parameters i and j denoting the size of the matrix.

      This is the same as \e putScalar<\c i,\c j>(0.0, \e mat), but it should be faster.

      \param mat
        matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void clearMatrix(CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      mat.putScalar(0.0);
    }

    /// Fill matrix with scalar value
    /*!
      Set every number in \e mat to \e scalar. This function takes two template
      parameters \c i and \c j denoting the size of the matrix.

      \param scalar
        scalar value to be set
      \param mat
        matrix, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void putScalar(const value_type scalar, CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      mat.putScalar(scalar);
    }

    /// Calculate absolut values of a matrix
    /*!
      Fill \e out with the absolute values from \e in. This function takes two
      template parameters \c i and \c j denoting the sizes of the matrices.

      \param out
        matrix to be set, size (\c i)x(\c j)
      \param in
        matrix the values are read from, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void abs(CORE::LINALG::SerialDenseMatrix::Base& dest,
        const CORE::LINALG::SerialDenseMatrix::Base& src);

    /// Calculate reciprocal values of a matrix
    /*!
      Fill \e out with the reciprocal of the values from \e in. This
      function takes two template parameters \c i and \c j denoting the
      sizes of the matrices.

      \param out
        matrix to be set, size (\c i)x(\c j)
      \param in
        matrix the values are read from, size (\c i)x(\c j)
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline void reciprocal(CORE::LINALG::SerialDenseMatrix::Base& dest,
        const CORE::LINALG::SerialDenseMatrix::Base& src);

    /// 1-norm
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return 1-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm1(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return norm1<value_type, i, j>(mat.values());
    }

    /// 2-norm (Euclidean norm)
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return 2-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm2(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return norm2<value_type, i, j>(mat.values());
    }

    /// Inf-norm
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return inf-norm of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type normInf(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return normInf<value_type, i, j>(mat.values());
    }

    /// Minimum value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return minimum value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type minValue(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return minValue<value_type, i, j>(mat.values());
    }

    /// Maximum value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return maximum value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type maxValue(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return maxValue<value_type, i, j>(mat.values());
    }

    /// Mean value of a matrix
    /*!
      The template arguments \c i and \c j are the size of the matrix.

      \param mat
        matrix, size (\c i)x(\c j)
      \return mean value of \e mat
     */
    template <class value_type, unsigned int i, unsigned int j>
    inline value_type meanValue(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
      return meanValue<value_type, i, j>(mat.values());
    }

    /*
     * Definitions of the functions taking value_type*
     *
     */

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiply(
        value_type_out* out, const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyNN(
        value_type_out* out, const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyNT(
        value_type_out* out, const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
          }
          *out = tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyTN(
        value_type_out* out, const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3];
          }
          *out = tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_left, class value_type_right>
    inline void multiplyTT(
        value_type_out* out, const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3 * k];
          }
          *out = tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiply(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyNN(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyNT(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
          }
          *out = infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyTN(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3];
          }
          *out = infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_infac, class value_type_left, class value_type_right>
    inline void multiplyTT(value_type_out* out, const value_type_infac infac,
        const value_type_left* const left, const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3 * k];
          }
          *out = infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiply(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = (*out) * outfac + infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyNN(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3];
          }
          *out = (*out) * outfac + infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyNT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i; ++c2)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3 * i] * right[c1 + c3 * k];
          }
          *out = (*out) * outfac + infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyTN(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < j * k; c1 += j)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3];
          }
          *out = (*out) * outfac + infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, unsigned int k,
        class value_type_outfac, class value_type_infac, class value_type_left,
        class value_type_right>
    inline void multiplyTT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_left* const left,
        const value_type_right* const right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      for (unsigned int c1 = 0; c1 < k; ++c1)
      {
        for (unsigned int c2 = 0; c2 < i * j; c2 += j)
        {
          value_type_out tmp = left[c2] * right[c1];
          for (unsigned int c3 = 1; c3 < j; ++c3)
          {
            tmp += left[c2 + c3] * right[c1 + c3 * k];
          }
          *out = (*out) * outfac + infac * tmp;
          ++out;
        }
      }
    }

    template <class value_type>
    inline value_type invert1x1(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      const value_type det = in[0];
      if (det == 0.0) FOUR_C_THROW("Determinant of 1x1 matrix is zero");
      out[0] = 1.0 / in[0];
      return det;
    }

    template <class value_type>
    inline value_type invert2x2(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      const value_type det = in[0] * in[1 + 1 * 2] - in[1] * in[1 * 2];
      if (det == 0.0) FOUR_C_THROW("Determinant of 2x2 matrix is zero");
      const value_type invdet = 1.0 / det;
      out[0] = invdet * in[1 + 1 * 2];
      out[1] = -invdet * in[1];
      out[1 * 2] = -invdet * in[1 * 2];
      out[1 + 1 * 2] = invdet * in[0];
      return det;
    }


    template <class value_type>
    inline value_type invert3x3(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      out[0] = in[1 + 1 * 3] * in[2 + 2 * 3] - in[2 + 1 * 3] * in[1 + 2 * 3];
      out[1] = in[2] * in[1 + 2 * 3] - in[1] * in[2 + 2 * 3];
      out[2] = in[1] * in[2 + 1 * 3] - in[2] * in[1 + 1 * 3];
      const value_type det = in[0] * out[0] + in[1 * 3] * out[1] + in[2 * 3] * out[2];
      // const value_type det = in[0]*in[1+3*1]*in[2+3*2] +
      //                   in[0+3*1]*in[1+3*2]*in[2+3*0] +
      //                   in[0+3*2]*in[1+3*0]*in[2+3*1] -
      //                   in[0+3*2]*in[1+3*1]*in[2+3*0] -
      //                   in[0+3*0]*in[1+3*2]*in[2+3*1] -
      //                   in[0+3*1]*in[1+3*0]*in[2+3*2];
      if (det == 0.0) FOUR_C_THROW("Determinant of 3x3 matrix is zero");
      const value_type invdet = 1.0 / det;
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

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      DENSEERROR::Checker<i == j>::Cannot_compute_inverse_of_nonsquare_matrix();

      switch (i)
      {
        case 1:
          return invert1x1(out, in);
        case 2:
          return invert2x2(out, in);
        case 3:
          return invert3x3(out, in);
        default:
          DENSEERROR::Checker<(
              i < 4)>::Use_FixedSizeSerialDenseSolver_for_matrices_bigger_than_3x3();
          return 0.0;
      }
    }


    template <class value_type>
    inline value_type invert1x1(value_type* mat)
    {
      const value_type det = mat[0];
      if (det == 0.0) FOUR_C_THROW("Determinant of 1x1 matrix is zero");
      mat[0] = 1.0 / mat[0];
      return det;
    }

    template <class value_type>
    inline value_type invert2x2(value_type* mat)
    {
      value_type tmp;
      const value_type det = mat[0] * mat[1 + 1 * 2] - mat[1] * mat[1 * 2];
      if (det == 0.0) FOUR_C_THROW("Determinant of 2x2 matrix is zero");
      const value_type invdet = 1.0 / det;
      tmp = mat[0];
      mat[0] = invdet * mat[1 + 1 * 2];
      mat[1 + 1 * 2] = invdet * tmp;
      mat[1] *= -invdet;
      mat[1 * 2] *= -invdet;
      return det;
    }


    template <class value_type>
    inline value_type invert3x3(value_type* mat)
    {
      const value_type tmp00 = mat[1 + 1 * 3] * mat[2 + 2 * 3] - mat[2 + 1 * 3] * mat[1 + 2 * 3];
      const value_type tmp10 = mat[2] * mat[1 + 2 * 3] - mat[1] * mat[2 + 2 * 3];
      const value_type tmp20 = mat[1] * mat[2 + 1 * 3] - mat[2] * mat[1 + 1 * 3];
      const value_type det = mat[0] * tmp00 + mat[1 * 3] * tmp10 + mat[2 * 3] * tmp20;
      // const value_type det = mat[0+3*0]*mat[1+3*1]*mat[2+3*2] +
      //                    mat[0+3*1]*mat[1+3*2]*mat[2+3*0] +
      //                    mat[0+3*2]*mat[1+3*0]*mat[2+3*1] -
      //                    mat[0+3*2]*mat[1+3*1]*mat[2+3*0] -
      //                    mat[0+3*0]*mat[1+3*2]*mat[2+3*1] -
      //                    mat[0+3*1]*mat[1+3*0]*mat[2+3*2];
      if (det == 0.0) FOUR_C_THROW("Determinant of 3x3 matrix is zero");
      const value_type invdet = 1.0 / det;
      const value_type tmp01 = mat[1 * 3];
      const value_type tmp11 = mat[1 + 1 * 3];
      const value_type tmp12 = mat[1 + 2 * 3];
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


    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(value_type* mat)
    {
      DENSEERROR::Checker<i == j>::Cannot_compute_inverse_of_nonsquare_matrix();

      switch (i)
      {
        case 1:
          return invert1x1(mat);
        case 2:
          return invert2x2(mat);
        case 3:
          return invert3x3(mat);
        default:
          DENSEERROR::Checker<(
              i < 4)>::Use_FixedSizeSerialDenseSolver_for_matrices_bigger_than_3x3();
          return 0.0;
      }
    }

    template <class value_type>
    inline value_type determinant_large_matrix(
        unsigned int i, unsigned int j, const value_type* mat)
    {
      FOUR_C_THROW("determinant_large_matrix not implemented for this value_type!");
      return 0.0;
    }

    // specialization for double as lapack routine is used
    template <>
    inline double determinant_large_matrix<double>(
        unsigned int i, unsigned int j, const double* mat)
    {
      // taken from src/linalg/linalg_utils_densematrix_eigen.cpp: CORE::LINALG::DeterminantLU,
      // only with minor changes.
      std::vector<double> tmp(i * j);
      std::copy(mat, mat + i * j, tmp.data());
      std::vector<int> ipiv(j);
      int info;

      Teuchos::LAPACK<int, double> lapack;
      lapack.GETRF(i, j, tmp.data(), i, ipiv.data(), &info);

      if (info < 0)
        FOUR_C_THROW("Lapack's dgetrf returned %d", info);
      else if (info > 0)
        return 0.0;
      double d = tmp[0];
      for (unsigned int c = 1; c < j; ++c) d *= tmp[c + i * c];
      // swapping rows of A changes the sign of the determinant, so we have to
      // undo lapack's permutation w.r.t. the determinant
      // note the fortran indexing convention in ipiv
      for (unsigned int c = 0; c < j; ++c)
        if (static_cast<unsigned>(ipiv[c]) != c + 1) d *= -1.0;
      return d;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type determinant(const value_type* mat)
    {
      DENSEERROR::Checker<i == j>::Cannot_compute_determinant_of_nonsquare_matrix();

      switch (i)
      {
        case 1:
          return *mat;
        case 2:
          return mat[0] * mat[1 + 1 * 2] - mat[1] * mat[1 * 2];
        case 3:
          return mat[0] * (mat[1 + 1 * 3] * mat[2 + 2 * 3] - mat[2 + 1 * 3] * mat[1 + 2 * 3]) +
                 mat[1 * 3] * (mat[2] * mat[1 + 2 * 3] - mat[1] * mat[2 + 2 * 3]) +
                 mat[2 * 3] * (mat[1] * mat[2 + 1 * 3] - mat[2] * mat[1 + 1 * 3]);
        default:
          return determinant_large_matrix<value_type>(i, j, mat);
      }
    }


    /* add matrices */

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_in>
    inline void update(value_type_out* out, const value_type_in* in)
    {
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
      {
#ifdef FOUR_C_ENABLE_ASSERTIONS
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
        // std::memcpy(out, in, i*j*sizeof(value_type));
        std::copy(in, in + i * j, out);
      }
      else
      {
        update<value_type_out, i, j>(out, 1.0, in);
      }
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_infac,
        class value_type_in>
    inline void update(value_type_out* out, const value_type_infac infac, const value_type_in* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out = infac * (*in);
      for (unsigned int c = 1; c < i * j; ++c) *(++out) = infac * (*(++in));
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_infac, class value_type_in>
    inline void update(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_in* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      if (outfac > -1e-30 and outfac < 1e-30)
      {  // cannot handle this case here, because 0*nan==nan
        update<value_type_out, i, j>(out, infac, in);
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

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_left,
        class value_type_right>
    inline void update(
        value_type_out* out, const value_type_left* left, const value_type_right* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = *left + *right;
      for (unsigned int c = 1; c < i * j; ++c) *(++out) = *(++left) + *(++right);
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_leftfac,
        class value_type_left, class value_type_rightfac, class value_type_right>
    inline void update(value_type_out* out, const value_type_leftfac leftfac,
        const value_type_left* left, const value_type_rightfac rightfac,
        const value_type_right* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = leftfac * (*left) + rightfac * (*right);
      for (unsigned int c = 1; c < i * j; ++c)
        *(++out) = leftfac * (*(++left)) + rightfac * (*(++right));
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_leftfac, class value_type_left, class value_type_rightfac,
        class value_type_right>
    inline void update(const value_type_outfac outfac, value_type_out* out,
        const value_type_leftfac leftfac, const value_type_left* left,
        const value_type_rightfac rightfac, const value_type_right* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_left>)
        if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if constexpr (std::is_same_v<value_type_out, value_type_right>)
        if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      if (outfac > -1e-30 and outfac < 1e-30)
      {  // cannot handle this case here, because 0*nan==nan
        update<value_type_out, i, j>(out, leftfac, left, rightfac, right);
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

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_in>
    inline void updateT(value_type_out* out, const value_type_in* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      for (unsigned int c2 = 0; c2 < j; c2 += 1)
        for (unsigned int c1 = 0; c1 < i; c1 += 1) *(out++) = in[c2 + c1 * j];
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_infac,
        class value_type_in>
    inline void updateT(value_type_out* out, const value_type_infac infac, const value_type_in* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      for (unsigned int c2 = 0; c2 < j; c2 += 1)
        for (unsigned int c1 = 0; c1 < i; c1 += 1) *(out++) = infac * in[c2 + c1 * j];
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_outfac,
        class value_type_infac, class value_type_in>
    inline void updateT(const value_type_outfac outfac, value_type_out* out,
        const value_type_infac infac, const value_type_in* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if constexpr (std::is_same_v<value_type_out, value_type_in>)
        if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      if (outfac > -1e-30 and outfac < 1e-30)
      {  // cannot handle this case here, because 0*nan==nan
        updateT<value_type_out, i, j>(out, infac, in);
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

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out *= *in;
      for (unsigned c = 1; c < i * j; ++c) *(++out) *= *(++in);
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type fac, value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out *= fac * (*in);
      for (unsigned c = 1; c < i * j; ++c) *(++out) *= fac * (*(++in));
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(value_type* out, const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = (*left) * (*right);
      for (unsigned c = 1; c < i * j; ++c) *(++out) = (*(++left)) * (*(++right));
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(
        value_type* out, const value_type infac, const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = infac * (*left) * (*right);
      for (unsigned c = 1; c < i * j; ++c) *(++out) = infac * (*(++left)) * (*(++right));
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type outfac, value_type* out, const value_type infac,
        const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      if (outfac > -1e-30 and outfac < 1e-30)
      {
        eMultiply<value_type, i, j>(out, infac, left, right);
        return;
      }
      *out = outfac * (*out) + infac * (*left) * (*right);
      for (unsigned c = 1; c < i * j; ++c)
      {
        ++out;
        *out = outfac * (*out) + infac * (*(++left)) * (*(++right));
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out /= *in;
      for (unsigned c = 1; c < i * j; ++c) *(++out) /= *(++in);
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type fac, value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out = fac * (*out) / (*in);
      for (unsigned c = 1; c < i * j; ++c)
      {
        ++out;
        ++in;
        *out = fac * (*out) / (*in);
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(value_type* out, const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = (*left) / (*right);
      for (unsigned c = 1; c < i * j; ++c) *(++out) = (*(++left)) / (*(++right));
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(
        value_type* out, const value_type infac, const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      *out = infac * (*left) / (*right);
      for (unsigned c = 1; c < i * j; ++c) *(++out) = infac * (*(++left)) / (*(++right));
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type outfac, value_type* out, const value_type infac,
        const value_type* left, const value_type* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == left) FOUR_C_THROW("'out' and 'left' point to same memory location");
      if (out == right) FOUR_C_THROW("'out' and 'right' point to same memory location");
#endif
      if (outfac > -1e-30 and outfac < 1e-30)
      {
        eDivide<value_type, i, j>(out, infac, left, right);
        return;
      }
      *out = outfac * (*out) + infac * (*left) / (*right);
      for (unsigned c = 1; c < i * j; ++c)
      {
        ++out;
        *out = outfac * (*out) + infac * (*(++left)) / (*(++right));
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void scaleMatrix(const value_type factor, value_type* mat)
    {
      *mat *= factor;
      for (unsigned int c = 1; c < i * j; ++c) *(++mat) *= factor;
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_left,
        class value_type_right>
    inline value_type_out dot(const value_type_left* left, const value_type_right* right)
    {
      value_type_out res = (*left) * (*right);
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++left;
        ++right;
        res += (*left) * (*right);
      }
      return res;
    }

    template <class value_type_out, unsigned int i, unsigned int j, class value_type_left,
        class value_type_right>
    inline void crossproduct(
        value_type_out* out, const value_type_left* left, const value_type_right* right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (i != 3 || j != 1) FOUR_C_THROW("cross product only for 3x1 matrices available");
#endif
      out[0] = left[1] * right[2] - left[2] * right[1];
      out[1] = left[2] * right[0] - left[0] * right[2];
      out[2] = left[0] * right[1] - left[1] * right[0];
      return;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void clearMatrix(value_type* mat)
    {
      // the memset method is needed for arbitrary precision (cln) data types instead of the fill
      // method std::memset(mat,0,i*j*sizeof(value_type));
      std::fill(mat, mat + i * j, 0.0);
    }


    template <class value_type, unsigned int i, unsigned int j>
    inline void putScalar(const value_type scalar, value_type* mat)
    {
      *mat = scalar;
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        *mat = scalar;
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void abs(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out = *in >= 0 ? *in : -*in;
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++out;
        ++in;
        *out = *in >= 0 ? *in : -*in;
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void reciprocal(value_type* out, const value_type* in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out == in) FOUR_C_THROW("'out' and 'in' point to same memory location");
#endif
      *out = 1.0 / (*in);
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++out;
        ++in;
        *out = 1.0 / (*in);
      }
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm1(const value_type* mat)
    {
      value_type result = *mat >= 0 ? *mat : -(*mat);
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        result += *mat >= 0 ? *mat : -(*mat);
      }
      return result;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type norm2(const value_type* mat)
    {
      value_type result = (*mat) * (*mat);
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        result += (*mat) * (*mat);
      }
      return CORE::MathOperations<value_type>::sqrt(result);
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type normInf(const value_type* mat)
    {
      value_type result = CORE::MathOperations<value_type>::abs(*mat);
      value_type tmp;
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        tmp = CORE::MathOperations<value_type>::abs(*mat);
        result = std::max(result, tmp);
      }
      return result;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type minValue(const value_type* mat)
    {
      value_type result = *mat;
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        if (*mat < result) result = *mat;
      }
      return result;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type maxValue(const value_type* mat)
    {
      value_type result = *mat;
      for (unsigned int c = 1; c < i * j; ++c)
      {
        ++mat;
        if (*mat > result) result = *mat;
      }
      return result;
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type meanValue(const value_type* mat)
    {
      value_type result = *mat;
      for (unsigned int c = 1; c < i * j; ++c) result += *(++mat);
      return result / (i * j);
    }

    /*
     * Definitions of the functions taking CORE::LINALG::SerialDenseMatrix::Base
     *
     */

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiply<value_type, i, j, k>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNN<value_type, i, j, k>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numRows() or
          left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNT<value_type, i, j, k>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numCols() or
          left.numRows() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTN<value_type, i, j, k>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numRows() or
          left.numRows() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTT<value_type, i, j, k>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiply<value_type, i, j, k>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNN<value_type, i, j, k>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numRows() or
          left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNT<value_type, i, j, k>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numCols() or
          left.numRows() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTN<value_type, i, j, k>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numRows() or
          left.numRows() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTT<value_type, i, j, k>(out.values(), infac, left.values(), right.values());
    }


    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiply(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiply<value_type, i, j, k>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNN(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numCols() or
          left.numCols() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNN<value_type, i, j, k>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyNT(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or out.numCols() != right.numRows() or
          left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i) * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyNT<value_type, i, j, k>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTN(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numCols() or
          left.numRows() != right.numRows())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTN<value_type, i, j, k>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j, unsigned int k>
    inline void multiplyTT(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numCols() or out.numCols() != right.numRows() or
          left.numRows() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for multiplication, (%i,%i) = (%i,%i)^T * (%i,%i)^T",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      multiplyTT<value_type, i, j, k>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type invert(
        CORE::LINALG::SerialDenseMatrix::Base& out, const CORE::LINALG::SerialDenseMatrix::Base& in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != out.numCols() or in.numRows() != in.numCols() or
          out.numRows() != in.numRows())
        FOUR_C_THROW("Invalid matrix sizes for inversion, (%i,%i) = inv( (%i,%i) )", out.numRows(),
            out.numCols(), in.numRows(), in.numCols());
#endif
      return invert<value_type, i, j>(out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline value_type determinant(const CORE::LINALG::SerialDenseMatrix::Base& mat)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (mat.numRows() != mat.numCols())
        FOUR_C_THROW(
            "Invalid matrix sizes for determinant, inv( (%i,%i) )", mat.numRows(), mat.numCols());
#endif
      return determinant<value_type, i, j>(mat.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out,
        const CORE::LINALG::SerialDenseMatrix::Base& left,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or left.numRows() != right.numRows() or
          out.numCols() != left.numCols() or left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) = (%i,%i) + (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      update<value_type, i, j>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type leftfac,
        const CORE::LINALG::SerialDenseMatrix::Base& left, const value_type rightfac,
        const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or left.numRows() != right.numRows() or
          out.numCols() != left.numCols() or left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) = (%i,%i) + (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      update<value_type, i, j>(out.values(), leftfac, left.values(), rightfac, right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void update(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type leftfac, const CORE::LINALG::SerialDenseMatrix::Base& left,
        const value_type rightfac, const CORE::LINALG::SerialDenseMatrix::Base& right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != left.numRows() or left.numRows() != right.numRows() or
          out.numCols() != left.numCols() or left.numCols() != right.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) = (%i,%i) + (%i,%i)",
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      update<value_type, i, j>(
          outfac, out.values(), leftfac, left.values(), rightfac, right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void update(
        CORE::LINALG::SerialDenseMatrix::Base& out, const CORE::LINALG::SerialDenseMatrix::Base& in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != in.numRows() or out.numCols() != in.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) += (%i,%i)", out.numRows(),
            out.numCols(), in.numRows(), in.numCols());
#endif
      update<value_type, i, j>(out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void update(CORE::LINALG::SerialDenseMatrix::Base& out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base& in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != in.numRows() or out.numCols() != in.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) += (%i,%i)", out.numRows(),
            out.numCols(), in.numRows(), in.numCols());
#endif
      update<value_type, i, j>(out.values(), infac, in.values());
    }


    template <class value_type, unsigned int i, unsigned int j>
    inline void update(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base& out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base& in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != in.numRows() or out.numCols() != in.numCols())
        FOUR_C_THROW("Invalid matrix sizes for addition, (%i,%i) += (%i,%i)", out.numRows(),
            out.numCols(), in.numRows(), in.numCols());
#endif
      update<value_type, i, j>(outfac, out.values(), infac, in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(
        CORE::LINALG::SerialDenseMatrix::Base out, const CORE::LINALG::SerialDenseMatrix::Base in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or in.numRows() != i or in.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eMultiply<%i,%i>, (%i,%i) *= (%i,%i)", i, j,
            out.numRows(), out.numCols(), in.numRows(), in.numCols());
#endif
      eMultiply<value_type, i, j>(out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type fac, CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or in.numRows() != i or in.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eMultiply<%i,%i>, (%i,%i) *= (%i,%i)", i, j,
            out.numRows(), out.numCols(), in.numRows(), in.numCols());
#endif
      eMultiply<value_type, i, j>(fac, out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eMultiply<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eMultiply<value_type, i, j>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(CORE::LINALG::SerialDenseMatrix::Base out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eMultiply<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eMultiply<value_type, i, j>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eMultiply(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eMultiply<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eMultiply<value_type, i, j>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(
        CORE::LINALG::SerialDenseMatrix::Base out, const CORE::LINALG::SerialDenseMatrix::Base in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or in.numRows() != i or in.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eDivide<%i,%i>, (%i,%i) *= (%i,%i)", i, j,
            out.numRows(), out.numCols(), in.numRows(), in.numCols());
#endif
      eDivide<value_type, i, j>(out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type fac, CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base in)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or in.numRows() != i or in.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eDivide<%i,%i>, (%i,%i) *= (%i,%i)", i, j,
            out.numRows(), out.numCols(), in.numRows(), in.numCols());
#endif
      eDivide<value_type, i, j>(fac, out.values(), in.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(CORE::LINALG::SerialDenseMatrix::Base out,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eDivide<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eDivide<value_type, i, j>(out.values(), left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(CORE::LINALG::SerialDenseMatrix::Base out, const value_type infac,
        const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eDivide<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eDivide<value_type, i, j>(out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void eDivide(const value_type outfac, CORE::LINALG::SerialDenseMatrix::Base out,
        const value_type infac, const CORE::LINALG::SerialDenseMatrix::Base left,
        const CORE::LINALG::SerialDenseMatrix::Base right)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (out.numRows() != i or out.numCols() != j or left.numRows() != i or left.numCols() != j or
          right.numRows() != i or right.numCols() != j)
        FOUR_C_THROW("Invalid matrix sizes in eDivide<%i,%i>, (%i,%i) = (%i,%i)*(%i,%i)", i, j,
            out.numRows(), out.numCols(), left.numRows(), left.numCols(), right.numRows(),
            right.numCols());
#endif
      eDivide<value_type, i, j>(outfac, out.values(), infac, left.values(), right.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void abs(CORE::LINALG::SerialDenseMatrix::Base& dest,
        const CORE::LINALG::SerialDenseMatrix::Base& src)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (dest.numRows() != dest.numRows() or src.numCols() != src.numCols())
        FOUR_C_THROW("Invalid matrix sizes for abs, (%i,%i) = abs( (%i,%i) )", dest.numRows(),
            dest.numCols(), src.numRows(), src.numCols());
#endif
      abs<value_type, i, j>(dest.values(), src.values());
    }

    template <class value_type, unsigned int i, unsigned int j>
    inline void reciprocal(CORE::LINALG::SerialDenseMatrix::Base& dest,
        const CORE::LINALG::SerialDenseMatrix::Base& src)
    {
#ifdef FOUR_C_ENABLE_ASSERTIONS
      if (dest.numRows() != dest.numRows() or src.numCols() != src.numCols())
        FOUR_C_THROW("Invalid matrix sizes for reciprocal, (%i,%i) = reciprocal( (%i,%i) )",
            dest.numRows(), dest.numCols(), src.numRows(), src.numCols());
#endif
      reciprocal<value_type, i, j>(dest.values(), src.values());
    }
  }  // namespace DENSEFUNCTIONS


  /// Serial dense matrix with templated dimensions
  /*!
    A serial dense matrix with templated dimensions that is supposed to
    be fast and lightweight. The default scalar type is double.
    The value_type-array is allocated on the stack (small sizes, up to 512
    bytes or on the heap (larger sizes) and stored in
    column-major order, just like in CORE::LINALG::SerialDenseMatrix::Base.

    The interface is based on that of CORE::LINALG::SerialDenseMatrix::Base and
    Epetra_MultiVector. The whole View/Copy thing works a little
    different, though. See the appropriate functions for details.

    There is no operator[]. It behaves differently in
    CORE::LINALG::SerialDenseMatrix::Base and CORE::LINALG::SerialDenseVector::Base, and is not
    needed in either of them.
   */
  template <unsigned int rows, unsigned int cols, class value_type = double>
  class Matrix
  {
   private:
    /// threshold for when to allocate the memory instead of placing the matrix on the stack.
    /// set to 512 bytes (or 64 entries for double matrices).
    static constexpr bool allocatesmemory_ = rows * cols * sizeof(value_type) > 512;

    /// the pointer holding the data
    value_type* data_;

    /// for small sizes of the matrix, avoid expensive memory allocation by storing
    /// the matrix on the stack
    value_type datafieldsmall_[allocatesmemory_ ? 1 : rows * cols];

    /// whether we are a view to some other matrix
    bool isview_;

    /// only in combination with isview_. Pure read access to the underlying data.
    bool isreadonly_;

   public:
    typedef value_type scalar_type;

    /// Default constructor
    /*!
      Constructs a new Matrix and allocates the
      memory. If \e setzero==true it is filled with zeros, otherwise it
      is left uninitialized.

      \param setzero
        whether matrix should be initialised to zero
     */
    explicit Matrix(bool setzero = true);

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
    explicit Matrix(value_type* d, bool view = false);

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
    explicit Matrix(const value_type* d, bool view = false);

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
    explicit Matrix(CORE::LINALG::SerialDenseMatrix::Base& d, bool view = false);

    /// Constructor
    /*!
      Constructs a new Matrix from data \e d. The data is copied.

      \param d
        matrix to be copied
     */
    explicit Matrix(const CORE::LINALG::SerialDenseMatrix::Base& d);

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
    Matrix(Matrix<rows, cols, value_type>& source, bool view);

    /// Copy constructor
    /*!
      Constructs a new Matrix from source. Unlike
      the CORE::LINALG::SerialDenseMatrix::Base copy constructor this one *always*
      copies the data, even when \e source is a view.

      \param source
        matrix to copy
     */
    Matrix(const Matrix<rows, cols, value_type>& source);

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
    Matrix(const Matrix<rows, cols, value_type>& source, bool view);

    /// Deconstructor
    ~Matrix();

    /// Return the value_type* holding the data.
    inline const value_type* A() const { return data_; }
    /// Return the value_type* holding the data.
    inline const value_type* values() const { return data_; }
    /// Return the value_type* holding the data.
    inline value_type* A()
    {
      FOUR_C_ASSERT((not isreadonly_), "No write access to read-only data!");
      return data_;
    }
    /// Return the value_type* holding the data.
    inline value_type* values()
    {
      FOUR_C_ASSERT((not isreadonly_), "No write access to read-only data!");
      return data_;
    }
    /// Return the number of rows
    static constexpr unsigned int M() { return numRows(); }
    /// Return the number of columns
    static constexpr unsigned int N() { return numCols(); }
    /// Return the number of rows
    static constexpr unsigned int numRows() { return rows; }
    /// Return the number of columns
    static constexpr unsigned int numCols() { return cols; }
    /// Check whether the matrix is initialized
    /*!
      You cannot test whether the matrix is empty using M() and N(),
      for they will always return the templated size. Instead this
      function can be used, it tests whether the data pointer is not
      nullptr.

      \note To actually get a matrix for which IsInitialized() returns
      false you must construct a view to nullptr, because the default
      constructor already allocates memory.
     */
    inline bool IsInitialized() const { return A() != nullptr; }

    /// Set view
    /*!
      Set this matrix to be a view to \e data.

      \param data
        memory to be viewed
     */
    void SetView(value_type* data);

    /// Set view
    /*!
      Set this matrix to be a view to \e source.

      \param source
        matrix to be viewed
     */
    void SetView(Matrix<rows, cols, value_type>& source);

    /// Set copy
    /*!
      Set this matrix to be a copy of \e data. The difference to Update(\e data)
      is that this funcion will allocate it's own memory when it was a
      view before, Update would copy the data into the view.

      \param data
        memory to copy
     */
    void SetCopy(const value_type* data);

    /// Set copy
    /*!
      Set this matrix to be a copy of source. Only the value_type array
      will be copied, the \e isview_ flag is ignored (this is equivalent to
      SetCopy(source.values()). The difference to Update(\e source) is that this funcion will
      allocate it's own memory when it was a view before, Update would copy the data into the view.

      \param source
        matrix to copy from
     */
    void SetCopy(const Matrix<rows, cols, value_type>& source);


    /// Calculate determinant
    /*!
      \return determinant
     */
    inline value_type Determinant() const;

    /// Invert in place
    /*!
      Invert this matrix in place.

      \return determinant of matrix before inversion
     */
    inline value_type Invert();

    /// Invert matrix
    /*!
      Invert matrix \e other and store the result in \e this.

      \param other
        matrix to be inverted
      \return determinant of \e other
     */
    inline value_type Invert(const Matrix<rows, cols, value_type>& other);

    /// Set to zero
    /*!
      Sets every value in this matrix to zero. This is equivalent to
      PutScalar(0.0), but it should be faster.
     */
    inline void Clear() { DENSEFUNCTIONS::clearMatrix<value_type, rows, cols>(A()); }

    // Epetra-style Functions
    /// Fill with scalar
    /*!
      Sets every value in this matrix to \e scalar.

      \param scalar
        value to fill matrix with
     */
    inline void PutScalar(const value_type scalar)
    {
      DENSEFUNCTIONS::putScalar<value_type, rows, cols>(scalar, A());
    }

    // Teuchos-style Functions
    /// Fill with scalar
    /*!
      Sets every value in this matrix to \e scalar.

      \param scalar
        value to fill matrix with
     */
    inline void putScalar(const value_type scalar)
    {
      DENSEFUNCTIONS::putScalar<value_type, rows, cols>(scalar, A());
    }

    /// Dot product
    /*!
      Return the dot product of \e this and \e other.

      \param other
        second factor
      \return dot product
     */
    template <class value_type_out = value_type, class value_type_other>
    inline value_type_out Dot(const Matrix<rows, cols, value_type_other>& other) const
    {
      return DENSEFUNCTIONS::dot<value_type_out, rows, cols>(A(), other.values());
    }

    /// Cross product
    /*!
      Return the cross product of \e left and \e right.

      \param left
      \param right
      \return cross product
     */
    template <class value_type_left, class value_type_right>
    inline void CrossProduct(const Matrix<rows, cols, value_type_left>& left,
        const Matrix<rows, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::crossproduct<value_type, rows, cols>(A(), left.values(), right.values());
    }

    /// Compute absolute value
    /*!
      Fill this matrix with the absolute value of the numbers in \e other.

      \param other
        matrix to read values from
     */
    inline void Abs(const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::abs<value_type, rows, cols>(A(), other.values());
    }

    /// Compute reciprocal value
    /*!
      Fill this matrix with the reciprocal value of the numbers in \e other.

      \param other
        matrix to read values from
     */
    inline void Reciprocal(const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::reciprocal<value_type, rows, cols>(A(), other.values());
    }

    /// Scale
    /*!
      Scale matrix with \e scalar.

      \param scalar
        scaling factor
     */
    inline void Scale(const value_type scalar)
    {
      DENSEFUNCTIONS::scaleMatrix<value_type, rows, cols>(scalar, A());
    }

    /// Copy: \e this = \e other
    /*!
      Copy \e other to \e this.

      \param other
        matrix to copy
     */
    template <class value_type_other>
    inline void Update(const Matrix<rows, cols, value_type_other>& other)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(A(), other.values());
    }

    /// Scaled copy: \e this = \e scalarOther * \e other
    /*!
      Copy \e scalarOther * \e other to \e this.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to read from
     */
    inline void Update(const value_type scalarOther, const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(A(), scalarOther, other.values());
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
    template <class value_type_scalar_other, class value_type_other, class value_type_scalar_this>
    inline void Update(const value_type_scalar_other scalarOther,
        const Matrix<rows, cols, value_type_other>& other, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(scalarThis, A(), scalarOther, other.values());
    }

    /// Add: \e this = \e left + \e right
    /*!
      Store \e left + \e right in this matrix.

      \param left
        first matrix to add
      \param right
        second matrix to add
     */
    template <class value_type_left, class value_type_right>
    inline void Update(const Matrix<rows, cols, value_type_left>& left,
        const Matrix<rows, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(A(), left.values(), right.values());
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
    template <class value_type_scalar_left, class value_type_left, class value_type_scalar_right,
        class value_type_right>
    inline void Update(const value_type_scalar_left scalarLeft,
        const Matrix<rows, cols, value_type_left>& left, const value_type_scalar_right scalarRight,
        const Matrix<rows, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(
          A(), scalarLeft, left.values(), scalarRight, right.values());
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
    template <class value_type_scalar_left, class value_type_left, class value_type_scalar_right,
        class value_type_right, class value_type_scalar_this>
    inline void Update(const value_type_scalar_left scalarLeft,
        const Matrix<rows, cols, value_type_left>& left, const value_type_scalar_right scalarRight,
        const Matrix<rows, cols, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(
          scalarThis, A(), scalarLeft, left.values(), scalarRight, right.values());
    }

    /// Transposed copy: \e this = \e other^T
    /*!
      Copy transposed \e other to \e this.

      \param other
        matrix to copy
     */
    template <class value_type_other>
    inline void UpdateT(const Matrix<cols, rows, value_type_other>& other)
    {
      DENSEFUNCTIONS::updateT<value_type, rows, cols>(A(), other.values());
    }

    /// Scaled transposed copy: \e this = \e scalarOther * \e other^T
    /*!
      Transposed copy \e scalarOther * \e other^T to \e this.

      \param scalarOther
        scaling factor for other
      \param other
        matrix to read from
     */
    template <class value_type_other_scalar, class value_type_other>
    inline void UpdateT(const value_type_other_scalar scalarOther,
        const Matrix<cols, rows, value_type_other>& other)
    {
      DENSEFUNCTIONS::updateT<value_type, rows, cols>(A(), scalarOther, other.values());
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
    template <class value_type_other_scalar, class value_type_other, class value_type_this_scalar>
    inline void UpdateT(const value_type_other_scalar scalarOther,
        const Matrix<cols, rows, value_type_other>& other, const value_type_this_scalar scalarThis)
    {
      DENSEFUNCTIONS::updateT<value_type, rows, cols>(scalarThis, A(), scalarOther, other.values());
    }

    /// Multiply element-wise: \e this(m,n) *= \e other(m,n)
    /*!
      Multiply \e this and \e other, storing the result in \e this.

      \param other
        factor
     */
    inline void EMultiply(const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::eMultiply<value_type, rows, cols>(A(), other.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e scalar * \e this(m,n)*\e other(m,n)
    /*!
      Multiply \e this and \e other, scale by \e scalar and store the result in \e this.

      \param scalar
        scaling factor for the product
      \param other
        factor
     */
    inline void EMultiply(const value_type scalar, const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::eMultiply<value_type, rows, cols>(scalar, A(), other.values());
    }

    /// Multiply element-wise: \e this(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Multiply \e left and \e right and store the result in \e this.

      \param left
        first factor
      \param right
        second factor
     */
    inline void EMultiply(
        const Matrix<rows, cols, value_type>& left, const Matrix<rows, cols, value_type>& right)
    {
      DENSEFUNCTIONS::eMultiply<value_type, rows, cols>(A(), left.values(), right.values());
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
    inline void EMultiply(const value_type scalarOther, const Matrix<rows, cols, value_type>& left,
        const Matrix<rows, cols, value_type>& right)
    {
      DENSEFUNCTIONS::eMultiply<value_type, rows, cols>(
          A(), scalarOther, left.values(), right.values());
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
    inline void EMultiply(const value_type scalarOther, const Matrix<rows, cols, value_type>& left,
        const Matrix<rows, cols, value_type>& right, const value_type scalarThis)
    {
      DENSEFUNCTIONS::eMultiply<value_type, rows, cols>(
          scalarThis, A(), scalarOther, left.values(), right.values());
    }

    /// Divide element-wise: \e this(m,n) *= \e other(m,n)
    /*!
      Divide \e this by \e other, storing the result in \e this.

      \param other
        factor
     */
    inline void EDivide(const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::eDivide<value_type, rows, cols>(A(), other.values());
    }

    /// Divide element-wise: \e this(m,n) = \e scalar * \e this(m,n)*\e other(m,n)
    /*!
      Divide \e this by \e other, scale by \e scalar and store the result in \e this.

      \param scalar
        scaling factor for the product
      \param other
        factor
     */
    inline void EDivide(const value_type scalar, const Matrix<rows, cols, value_type>& other)
    {
      DENSEFUNCTIONS::eDivide<value_type, rows, cols>(scalar, A(), other.values());
    }

    /// Divide element-wise: \e this(m,n) = \e left(m,n)*\e right(m,n)
    /*!
      Divide \e left by \e right and store the result in \e this.

      \param left
        dividend
      \param right
        divisor
     */
    inline void EDivide(
        const Matrix<rows, cols, value_type>& left, const Matrix<rows, cols, value_type>& right)
    {
      DENSEFUNCTIONS::eDivide<value_type, rows, cols>(A(), left.values(), right.values());
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
    inline void EDivide(const value_type scalarOther, const Matrix<rows, cols, value_type>& left,
        const Matrix<rows, cols, value_type>& right)
    {
      DENSEFUNCTIONS::eDivide<value_type, rows, cols>(
          A(), scalarOther, left.values(), right.values());
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
    inline void EDivide(const value_type scalarOther, const Matrix<rows, cols, value_type>& left,
        const Matrix<rows, cols, value_type>& right, const value_type scalarThis)
    {
      DENSEFUNCTIONS::eDivide<value_type, rows, cols>(
          scalarThis, A(), scalarOther, left.values(), right.values());
    }


    /// Calculate 1-norm
    /*!
      This is *not* the same as CORE::LINALG::SerialDenseMatrix::Base::NormOne.

      \return 1-norm
     */
    inline value_type Norm1() const { return DENSEFUNCTIONS::norm1<value_type, rows, cols>(A()); }

    /// Calculate 2-norm (Euclidean norm)
    /*!
      \return 2-norm
     */
    inline value_type Norm2() const { return DENSEFUNCTIONS::norm2<value_type, rows, cols>(A()); }

    /// Calculate inf-norm
    /*!
      This is *not* the same as CORE::LINALG::SerialDenseMatrix::Base::NormInf.

      \return inf-norm
     */
    inline value_type NormInf() const
    {
      return DENSEFUNCTIONS::normInf<value_type, rows, cols>(A());
    }

    /// Calculate minimum value
    /*!
      \return minimum value
     */
    inline value_type MinValue() const
    {
      return DENSEFUNCTIONS::minValue<value_type, rows, cols>(A());
    }

    /// Calculate maximum value
    /*!
      \return maximum value
     */
    inline value_type MaxValue() const
    {
      return DENSEFUNCTIONS::maxValue<value_type, rows, cols>(A());
    }

    /// Calculate mean value
    /*!
      \return mean value
     */
    inline value_type MeanValue() const
    {
      return DENSEFUNCTIONS::meanValue<value_type, rows, cols>(A());
    }

    /// Multiply: \e this = \e left*right
    /*!
      This is equivalent to MultiplyNN(\e left,\e right).

      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_left, class value_type_right>
    inline void Multiply(const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(A(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left*right
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_left, class value_type_right>
    inline void MultiplyNN(const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(A(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left*right^T
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_left, class value_type_right>
    inline void MultiplyNT(const Matrix<rows, inner, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyNT<value_type, rows, inner, cols>(A(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left^T*right
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_left, class value_type_right>
    inline void MultiplyTN(const Matrix<inner, rows, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyTN<value_type, rows, inner, cols>(A(), left.values(), right.values());
    }

    /// Multiply: \e this = \e left^T*right^T
    /*!
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_left, class value_type_right>
    inline void MultiplyTT(const Matrix<inner, rows, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyTT<value_type, rows, inner, cols>(A(), left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right>
    inline void Multiply(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(
          A(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarOthers * \e left*right
    /*!
      This is equivalent to MultiplyNN(\e scalarOthers,\e left,\e right).

      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
     */
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right>
    inline void MultiplyNN(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(
          A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right>
    inline void MultiplyNT(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyNT<value_type, rows, inner, cols>(
          A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right>
    inline void MultiplyTN(const value_type_scalar_other scalarOthers,
        const Matrix<inner, rows, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyTN<value_type, rows, inner, cols>(
          A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right>
    inline void MultiplyTT(const value_type_scalar_other scalarOthers,
        const Matrix<inner, rows, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right)
    {
      DENSEFUNCTIONS::multiplyTT<value_type, rows, inner, cols>(
          A(), scalarOthers, left.values(), right.values());
    }

    /// Multiply: \e this = \e scalarThis * \e this + \e scalarOthers * \e left*right
    /*!
      This is equivalent to MultiplyNN(\e scalarOthers,\e left,\e right,\e scalarThis).

      \param scalarOthers
        scalar factor for \e left*right
      \param left
        first factor
      \param right
        second factor
      \param scalarThis
        scalar factor for \e this
     */
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right, class value_type_scalar_this>
    inline void Multiply(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(
          scalarThis, A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right, class value_type_scalar_this>
    inline void MultiplyNN(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::multiply<value_type, rows, inner, cols>(
          scalarThis, A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right, class value_type_scalar_this>
    inline void MultiplyNT(const value_type_scalar_other scalarOthers,
        const Matrix<rows, inner, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::multiplyNT<value_type, rows, inner, cols>(
          scalarThis, A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right, class value_type_scalar_this>
    inline void MultiplyTN(const value_type_scalar_other scalarOthers,
        const Matrix<inner, rows, value_type_left>& left,
        const Matrix<inner, cols, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::multiplyTN<value_type, rows, inner, cols>(
          scalarThis, A(), scalarOthers, left.values(), right.values());
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
    template <unsigned int inner, class value_type_scalar_other, class value_type_left,
        class value_type_right, class value_type_scalar_this>
    inline void MultiplyTT(const value_type_scalar_other scalarOthers,
        const Matrix<inner, rows, value_type_left>& left,
        const Matrix<cols, inner, value_type_right>& right, const value_type_scalar_this scalarThis)
    {
      DENSEFUNCTIONS::multiplyTT<value_type, rows, inner, cols>(
          scalarThis, A(), scalarOthers, left.values(), right.values());
    }

    /// Write \e this to \e out
    /*!
      Write a readable representation of \e this to \e out. This function is called by
      \e out << *\e this.

      \param out
        out stream
     */
    void Print(std::ostream& out) const;

    /// = operator
    /*!
      Copy data from \e other to \e this, equivalent to Update(other).
      \param other
        matrix to get data from
     */
    inline Matrix<rows, cols, value_type>& operator=(const Matrix<rows, cols, value_type>& other);

    /// = operator for double
    /*!
      Fill with double \e other, same as PutScalar(other).

      \param other
        scalar value
     */
    inline Matrix<rows, cols, value_type>& operator=(const value_type other);

    /// == operator
    /*!
      Compare \e this with \e other.

      \param other
        matrix to compare with
     */
    inline bool operator==(const Matrix<rows, cols, value_type>& other) const;

    /// != operator
    /*!
      Compare \e this with \e other.

      \param other
        matrix to compare with
     */
    inline bool operator!=(const Matrix<rows, cols, value_type>& other) const;

    /// += operator
    /*!
      Add \e other to \e this.

      \param other
        matrix to add
     */
    template <class value_type_other>
    inline Matrix<rows, cols, value_type>& operator+=(
        const Matrix<rows, cols, value_type_other>& other)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(1.0, A(), 1.0, other.values());
      return *this;
    }

    /// -= operator
    /*!
      Subtract \e other from \e this.

      \param other
        matrix to subtract
     */
    template <class value_type_other>
    inline Matrix<rows, cols, value_type>& operator-=(
        const Matrix<rows, cols, value_type_other>& other)
    {
      DENSEFUNCTIONS::update<value_type, rows, cols>(1.0, A(), -1.0, other.values());
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
    inline value_type& operator()(unsigned int r, unsigned int c);

    /// Access data
    /*!
      Return value in row \e r and column \e c.

      \param r
        row index
      \param c
        column index
     */
    inline const value_type& operator()(unsigned int r, unsigned int c) const;

    /// Access data
    /*!
      Return value in row \e r. This works only for Matrices with cols==1 or rows==1 (vectors),
      otherwise a compile time error is raised.

      \param r
        index
     */
    inline value_type& operator()(unsigned int r);  // for vectors, with check at compile-time

    /// Access data
    /*!
      Return value in row \e r. This works only for Matrices with cols==1 or rows==1 (vectors),
      otherwise a compile time error is raised.

      \param r
        index
     */
    inline const value_type& operator()(unsigned int r) const;
  };

  template <class value_type, unsigned int cols, unsigned int rows>
  std::ostream& operator<<(std::ostream& out, const Matrix<rows, cols, value_type>& matrix);

  // Constructors

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(bool setzero)
      : data_(nullptr), isview_(false), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and (cols != 0)>::Matrix_dimensions_cannot_be_zero();
    if (allocatesmemory_)
      data_ = new value_type[rows * cols];
    else
      data_ = datafieldsmall_;
    if (setzero) DENSEFUNCTIONS::clearMatrix<value_type, rows, cols>(data_);
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(value_type* d, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (isview_)
    {
      data_ = d;
    }
    else
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(d, d + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(const value_type* d, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (isview_)
    {
      isreadonly_ = true;
      data_ = const_cast<value_type*>(d);
    }
    else
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(d, d + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(CORE::LINALG::SerialDenseMatrix::Base& d, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (d.values() == nullptr) return;
    if (d.numRows() != rows or d.numCols() != cols)
      FOUR_C_THROW("illegal matrix dimension (%d,%d)", d.numRows(), d.numCols());
    if (isview_)
    {
      data_ = d.values();
    }
    else
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(d.values(), d.values() + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(const CORE::LINALG::SerialDenseMatrix::Base& d)
      : Matrix(const_cast<CORE::LINALG::SerialDenseMatrix::Base&>(d), false)
  {
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(Matrix<rows, cols, value_type>& source, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (isview_)
    {
      data_ = source.data_;
    }
    else
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(source.data_, source.data_ + rows * cols, data_);
    }
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(const Matrix<rows, cols, value_type>& source)
      : data_(nullptr), isview_(false), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (allocatesmemory_)
      data_ = new value_type[rows * cols];
    else
      data_ = datafieldsmall_;
    std::copy(source.data_, source.data_ + rows * cols, data_);
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::Matrix(const Matrix<rows, cols, value_type>& source, bool view)
      : data_(nullptr), isview_(view), isreadonly_(false)
  {
    DENSEERROR::Checker<(rows != 0) and cols != 0>::Matrix_dimensions_cannot_be_zero();
    if (isview_)
    {
      isreadonly_ = true;
      data_ = const_cast<value_type*>(source.values());
    }
    else
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      std::copy(source.data_, source.data_ + rows * cols, data_);
    }
  }

  // Destructor
  template <unsigned int rows, unsigned int cols, class value_type>
  Matrix<rows, cols, value_type>::~Matrix()
  {
    if (allocatesmemory_ && not isview_) delete[] data_;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  void Matrix<rows, cols, value_type>::SetView(value_type* data)
  {
    if (not isview_)
    {
      if (allocatesmemory_) delete[] data_;
      isview_ = true;
    }
    data_ = data;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  void Matrix<rows, cols, value_type>::SetView(Matrix<rows, cols, value_type>& source)
  {
    if (not isview_)
    {
      if (allocatesmemory_) delete[] data_;
      isview_ = true;
    }
    data_ = source.data_;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  void Matrix<rows, cols, value_type>::SetCopy(const value_type* data)
  {
    if (isview_)
    {
      if (allocatesmemory_)
        data_ = new value_type[rows * cols];
      else
        data_ = datafieldsmall_;
      isview_ = false;
    }
    std::copy(data, data + rows * cols, data_);
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  void Matrix<rows, cols, value_type>::SetCopy(const Matrix<rows, cols, value_type>& source)
  {
    SetCopy(source.values());
  }

  // Determinant
  template <unsigned int rows, unsigned int cols, class value_type>
  inline value_type Matrix<rows, cols, value_type>::Determinant() const
  {
    DENSEERROR::Checker<rows == cols>::Cannot_compute_determinant_of_nonsquare_matrix();
    return DENSEFUNCTIONS::determinant<value_type, rows, cols>(A());
  }

  // Invert
  template <unsigned int rows, unsigned int cols, class value_type>
  inline value_type Matrix<rows, cols, value_type>::Invert()
  {
    DENSEERROR::Checker<rows == cols>::Cannot_compute_inverse_of_nonsquare_matrix();
    return DENSEFUNCTIONS::invert<value_type, rows, cols>(A());
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline value_type Matrix<rows, cols, value_type>::Invert(
      const Matrix<rows, cols, value_type>& other)
  {
    DENSEERROR::Checker<rows == cols>::Cannot_compute_inverse_of_nonsquare_matrix();
    return DENSEFUNCTIONS::invert<value_type, rows, cols>(A(), other.values());
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  void Matrix<rows, cols, value_type>::Print(std::ostream& out) const
  {
    out << "Matrix<" << rows << ',' << cols << '>';
    if (isview_) out << " (view to memory only)";
    if (isreadonly_) out << "\n (read only)";
    if (A() == nullptr)
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
          out << A()[i + rows * j];
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
        out << A()[i];
      }
      out << "]\n";
    }
  }

  /// output operator for Matrix
  /*!
    Write matrix to out. This function calls matrix.Print(out).
   */
  template <class value_type, unsigned int cols, unsigned int rows>
  std::ostream& operator<<(std::ostream& out, const Matrix<rows, cols, value_type>& matrix)
  {
    matrix.Print(out);
    return out;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline Matrix<rows, cols, value_type>& Matrix<rows, cols, value_type>::operator=(
      const Matrix<rows, cols, value_type>& other)
  {
    DENSEFUNCTIONS::update<value_type, rows, cols>(A(), other.values());
    return *this;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline Matrix<rows, cols, value_type>& Matrix<rows, cols, value_type>::operator=(
      const value_type other)
  {
    DENSEFUNCTIONS::putScalar<value_type, rows, cols>(other, A());
    return *this;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline bool Matrix<rows, cols, value_type>::operator==(
      const Matrix<rows, cols, value_type>& other) const
  {
    if (A() == other.values()) return true;
    // unfortunately memcmp does not work, because +0 and -0 are
    // different in memory...
    for (unsigned int c = 0; c < rows * cols; ++c)
      if (A()[c] != other.values()[c]) return false;
    return true;
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline bool Matrix<rows, cols, value_type>::operator!=(
      const Matrix<rows, cols, value_type>& other) const
  {
    return not(*this == other);
  }

  // Access operator

  template <unsigned int rows, unsigned int cols, class value_type>
  inline value_type& Matrix<rows, cols, value_type>::operator()(unsigned int r, unsigned int c)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= rows or c >= cols)
      FOUR_C_THROW("Indices %i,%i out of range in Matrix<%i,%i>.", r, c, rows, cols);
#endif
    return A()[r + c * rows];
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline const value_type& Matrix<rows, cols, value_type>::operator()(
      unsigned int r, unsigned int c) const
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= rows or c >= cols)
      FOUR_C_THROW("Indices %i,%i out of range in Matrix<%i,%i>.", r, c, rows, cols);
#endif
    return A()[r + c * rows];
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline value_type& Matrix<rows, cols, value_type>::operator()(unsigned int r)
  {
    DENSEERROR::Checker<(cols == 1) or (rows == 1)>::Cannot_call_1D_access_function_on_2D_matrix();
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= (cols == 1 ? rows : cols))
      FOUR_C_THROW("Index %i out of range in Matrix<%i,%i>.", r, rows, cols);
#endif
    return A()[r];
  }

  template <unsigned int rows, unsigned int cols, class value_type>
  inline const value_type& Matrix<rows, cols, value_type>::operator()(unsigned int r) const
  {
    DENSEERROR::Checker<(cols == 1) or (rows == 1)>::Cannot_call_1D_access_function_on_2D_matrix();
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (r >= (cols == 1 ? rows : cols))
      FOUR_C_THROW("Index %i out of range in Matrix<%i,%i>.", r, rows, cols);
#endif
    return A()[r];
  }


  /// A solver for fixed size serial dense matrices
  /*!
    This solver is intended to provide the funcionality of
    Epetra_SerialDenseSolver for fixed size matrices. So far only a
    subset (only the equilibration and transpose flags are available) is
    implemented for it is all that was needed. All the code of this
    solver is directly based on the Epetra solver, but with the attempt
    to simplify it and to avoid invalid states. This simplification
    might make it necessary to rewrite the class once more functionality
    is needed.

    The first two template argument specify the size of the matrix,
    although it is expected to be square. The third argument is the
    number of columns of the 'vectors'.

    \author Martin Kuettler
    \date 09/08
   */
  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs = 1>
  class FixedSizeSerialDenseSolver
  {
   private:
    // we do not need these functions
    FixedSizeSerialDenseSolver(const FixedSizeSerialDenseSolver<rows, cols, dim_rhs>&);
    FixedSizeSerialDenseSolver& operator=(const FixedSizeSerialDenseSolver<rows, cols, dim_rhs>&);

    /// wrapper for the LAPACK functions
    static Teuchos::LAPACK<int, double> lapack_;
    /// wrapper for the BLAS functions
    static Teuchos::BLAS<unsigned int, double> blas_;

    /// the matrix we got
    Matrix<rows, cols, double>* matrix_;
    /// the vector of unknowns
    Matrix<cols, dim_rhs, double>* vec_X_;
    /// the right hand side vector
    Matrix<rows, dim_rhs, double>* vec_B_;

    /// some storage for LAPACK
    std::vector<int> pivot_vec_;
    /// vector used for equilibration
    std::vector<double> r_;
    /// vector used for equilibration
    std::vector<double> c_;

    /// do we want to equilibrate?
    bool equilibrate_;
    /// should the matrix be used transposed?
    bool transpose_;
    /// is the matrix factored?
    bool factored_;
    /// is the matrix inverted?
    bool inverted_;
    /// is the system solved?
    bool solved_;


    /// Compute equilibrate scaling
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int ComputeEquilibrateScaling();

    /// Equilibrate matrix
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int EquilibrateMatrix();

    /// Equilibrate right hand side vector
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int EquilibrateRHS();

    /// Unequilibrate vector of unknowns
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int UnequilibrateLHS();

   public:
    /// Constructor
    FixedSizeSerialDenseSolver();

    /// Is matrix factored?
    /*!
      \return true if matrix is factored, false otherwise
     */
    bool IsFactored() { return factored_; }

    /// Is matrix inverted?
    /*!
      \return true if matrix is inverted, false otherwise
     */
    bool IsInverted() { return inverted_; }

    /// Is system solved?
    /*!
      \return true if system is solved, false otherwise
     */
    bool IsSolved() { return solved_; }

    /// Set the matrix
    /*!
      Set the matrix to mat.

      \param mat
        new matrix
     */
    void SetMatrix(Matrix<rows, cols, double>& mat);

    /// Set the vectors
    /*!
      Set the vectors, the new equation is matrix*X=B.

      \param X
        vector of unknowns
      \param B
        right hand side vector
     */
    void SetVectors(Matrix<cols, dim_rhs, double>& X, Matrix<rows, dim_rhs, double>& B);

    /// Set the equilibration
    /*!
      Set whether equilibration should be used.

      \param b
        new value for equilibrate_
     */
    void FactorWithEquilibration(bool b);

    /// Set transpose
    /*!
      Set whether the matrix should be used tranposed.

      \param b
        new value for transpose_
     */
    void SolveWithTranspose(bool b) { transpose_ = b; }

    /// Factor the matrix
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int Factor();

    /// Solve the system
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code or -100, indicating that
      the two vectors are the same, but may not be (when the matrix is
      inverted before the call to Solve).
     */
    int Solve();

    /// Invert the matrix
    /*
      \return integer error code. 0 if successful, negative
      otherwise. This is a LAPACK error code.
     */
    int Invert();
  };

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::FixedSizeSerialDenseSolver()
      : matrix_(nullptr),
        vec_X_(nullptr),
        vec_B_(nullptr),
        pivot_vec_(),
        r_(),
        c_(),
        equilibrate_(false),
        transpose_(false),
        factored_(false),
        inverted_(false),
        solved_(false)
  {
    DENSEERROR::Checker<(rows != 0) and (cols != 0)>::Matrix_dimensions_cannot_be_zero();
    DENSEERROR::Checker<rows == cols>::Matrix_size_in_solver_must_be_square();
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  void FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::SetMatrix(Matrix<rows, cols, double>& mat)
  {
    c_.clear();
    r_.clear();
    pivot_vec_.clear();
    inverted_ = factored_ = solved_ = false;
    // vec_B_ = vec_X_ = nullptr;
    matrix_ = &mat;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  void FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::SetVectors(
      Matrix<cols, dim_rhs, double>& X, Matrix<rows, dim_rhs, double>& B)
  {
    solved_ = false;
    vec_X_ = &X;
    vec_B_ = &B;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  void FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::FactorWithEquilibration(bool b)
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (factored_ or inverted_)
      FOUR_C_THROW("Cannot set equilibration after changing the matrix with Factor() or Invert().");
#endif
    equilibrate_ = b;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::Factor()
  {
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (inverted_) FOUR_C_THROW("Cannot factor the inverted matrix.");
#endif
    if (factored_) return 0;
    int errnum = 0;
    if (equilibrate_) errnum = EquilibrateMatrix();
    if (errnum != 0) return errnum;
    if (pivot_vec_.empty()) pivot_vec_.resize(rows);
    lapack_.GETRF(rows, cols, matrix_->A(), rows, pivot_vec_.data(), &errnum);
    if (errnum != 0) return errnum;

    factored_ = true;
    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::Solve()
  {
    int errnum = 0;
    if (equilibrate_)
    {
      errnum = EquilibrateRHS();
    }
    if (errnum != 0) return errnum;
#ifdef FOUR_C_ENABLE_ASSERTIONS
    if (not vec_B_ or not vec_X_) FOUR_C_THROW("Both vectors must be set to solve.");
#endif

    if (inverted_)
    {
      if (vec_B_ == vec_X_) return -100;

      blas_.GEMM(transpose_ ? Teuchos::TRANS : Teuchos::NO_TRANS, Teuchos::NO_TRANS, cols, dim_rhs,
          cols, 1.0, matrix_->A(), rows, vec_B_->A(), rows, 0.0, vec_X_->A(), cols);
      solved_ = true;
    }
    else
    {
      if (!factored_)
      {
        errnum = Factor();
        if (errnum != 0) return errnum;
      }

      if (vec_B_ != vec_X_) *vec_X_ = *vec_B_;
      lapack_.GETRS(transpose_ ? 'T' : 'N', cols, dim_rhs, matrix_->A(), rows, pivot_vec_.data(),
          vec_X_->A(), cols, &errnum);
      if (errnum != 0) return errnum;
      solved_ = true;
    }
    if (equilibrate_) errnum = UnequilibrateLHS();
    if (errnum != 0) return errnum;
    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::ComputeEquilibrateScaling()
  {
    if (!r_.empty()) return 0;  // we already did that
    int errnum;
    double rowcnd, colcnd, amax;
    r_.resize(rows);
    c_.resize(cols);
    lapack_.GEEQU(
        rows, cols, matrix_->A(), rows, r_.data(), c_.data(), &rowcnd, &colcnd, &amax, &errnum);
    if (errnum != 0) return errnum;

    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::EquilibrateMatrix()
  {
    int errnum = 0;
    if (r_.empty()) errnum = ComputeEquilibrateScaling();
    if (errnum != 0) return errnum;
    double* ptr = matrix_->A();
    double s1;
    for (unsigned j = 0; j < cols; ++j)
    {
      s1 = c_[j];
      for (unsigned i = 0; i < rows; ++i)
      {
        *ptr *= s1 * r_[i];
        ++ptr;
      }
    }
    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::EquilibrateRHS()
  {
    int errnum = 0;
    if (r_.empty()) errnum = ComputeEquilibrateScaling();
    if (errnum != 0) return errnum;
    std::vector<double>& r = transpose_ ? c_ : r_;
    double* ptr = vec_B_->A();
    for (unsigned j = 0; j < dim_rhs; ++j)
    {
      for (unsigned i = 0; i < cols; ++i)
      {
        *ptr *= r[i];
        ++ptr;
      }
    }

    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::UnequilibrateLHS()
  {
    std::vector<double>& c = transpose_ ? r_ : c_;
    double* ptr = vec_X_->A();
    for (unsigned j = 0; j < dim_rhs; ++j)
    {
      for (unsigned i = 0; i < rows; ++i)
      {
        *ptr *= c[i];
        ++ptr;
      }
    }

    return 0;
  }

  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  int FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::Invert()
  {
    int errnum;
    if (not factored_)
    {
      errnum = Factor();
      if (errnum != 0) return errnum;
    }

    int lwork = 4 * cols;
    std::vector<double> work(lwork);
    lapack_.GETRI(cols, matrix_->A(), rows, pivot_vec_.data(), work.data(), lwork, &errnum);
    if (errnum != 0) return errnum;
    inverted_ = true;
    factored_ = false;

    return 0;
  }

  // Initialize the static objects.
  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  Teuchos::LAPACK<int, double> FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::lapack_;
  template <unsigned int rows, unsigned int cols, unsigned int dim_rhs>
  Teuchos::BLAS<unsigned int, double> FixedSizeSerialDenseSolver<rows, cols, dim_rhs>::blas_;


}  // namespace CORE::LINALG

FOUR_C_NAMESPACE_CLOSE

#endif

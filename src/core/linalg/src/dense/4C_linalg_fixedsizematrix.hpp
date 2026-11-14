// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP
#define FOUR_C_LINALG_FIXEDSIZEMATRIX_HPP

#include "4C_config.hpp"

#include "4C_linalg_densefunctions.hpp"
#include "4C_utils_exceptions.hpp"

#include <ostream>
#include <span>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  class SerialDenseMatrix;
  class SerialDenseVector;

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
    Core::LinAlg::SerialDenseMatrix and Core::LinAlg::SerialDenseVector, and is not
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

    static constexpr size_t size_ = rows * cols;

    /// the pointer holding the data
    ValueType* data_;

    /// for small sizes of the matrix, avoid expensive memory allocation by storing
    /// the matrix on the stack
    ValueType datafieldsmall_[allocatesmemory_ ? 1 : size_];

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

    /// Return a std::span for easy access to the data as a 1D array.
    ///
    /// This is only available for vectors (rows==1 or cols==1).
    [[nodiscard]] std::span<const ValueType, size_> as_span() const
      requires(rows == 1 || cols == 1)
    {
      return std::span<ValueType, size_>(data_, size_);
    }

    /// Return a std::span for easy access to the data as a 1D array.
    ///
    /// This is only available for vectors (rows==1 or cols==1).
    [[nodiscard]] std::span<ValueType, size_> as_span()
      requires(rows == 1 || cols == 1)
    {
      FOUR_C_ASSERT((not isreadonly_), "No write access to read-only data!");
      return std::span<ValueType, size_>(data_, size_);
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

      \note This function will throw an exception if the matrix norm is zero.
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

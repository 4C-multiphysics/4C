// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FAD_HPP
#define FOUR_C_UTILS_FAD_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_mathoperations_sacado.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Core::FADUtils
{
  template <typename T>
  concept ExplicitlyConvertibleToDouble = requires(T a) {
    { static_cast<double>(a) };
  };

  /*!
   * \brief Overload of CastToDouble() for any type that is convertible to double.
   */
  template <ExplicitlyConvertibleToDouble ScalarType>
  inline double cast_to_double(ScalarType a)
  {
    return static_cast<double>(a);
  }

  /**
   * \brief Cast of FAD to double.
   *
   * This function will be used exclusively for FAD or FAD expression types and can also handle
   * nested FAD types, i.e., higher order derivatives.
   *
   * @note We recursively call this function with the value of the FAD variable @p a until we reach
   * the innermost double value.
   */
  template <typename FADType>
    requires(Sacado::IsFad<FADType>::value || Sacado::IsExpr<FADType>::value)
  inline double cast_to_double(FADType a)
  {
    return cast_to_double(a.val());
  }


  /*!
  \brief Cast of a FAD matrix to a double matrix
  */
  template <typename Type, unsigned int dim1, unsigned int dim2>
  Core::LinAlg::Matrix<dim1, dim2, double> cast_to_double(Core::LinAlg::Matrix<dim1, dim2, Type> a)
  {
    Core::LinAlg::Matrix<dim1, dim2, double> b(Core::LinAlg::Initialization::zero);

    for (unsigned int i = 0; i < dim1; i++)
    {
      for (unsigned int j = 0; j < dim2; j++)
      {
        b(i, j) = cast_to_double(a(i, j));
      }
    }
    return b;
  }

  /*!
  \brief Calculate signum function of FAD or double quantity
  */
  template <typename Type>
  double signum(Type a)
  {
    if (a >= 0.0)
      return 1.0;
    else
      return -1.0;
  }

  /*!
  \brief Calculate square root of a scalar FAD quantity. If a compiler Error is thrown when calling
  this function, check if the template argument is explicitly stated in the function call, i.e.
  Core::FADUtils::sqrt<my_AD_type>(...)
  */
  template <typename ScalarType>
  inline ScalarType sqrt(ScalarType a)
  {
    // avoid non-differentiable point of square root function by using conditional
    /* Todo should we use a tolerance/threshold here? */
    if (a == 0.0)
      return 0.0;
    else
      return std::sqrt(a);
  }

  /*!
  \brief Overload for double.
  */
  inline double sqrt(double a) { return std::sqrt(a); }

  /*!
  \brief Calculate Norm of a scalar FAD quantity. If a compiler Error is thrown when calling this
  function, check if the template argument is explicitly stated in the function call, i.e.
  Core::FADUtils::Norm<my_AD_type>(...)
  */
  template <typename ScalarType>
  inline ScalarType norm(ScalarType a)
  {
    return Core::FADUtils::sqrt<ScalarType>(a * a);
  }

  /*!
  \brief Calculate Norm of a FAD vector
  */
  template <typename ScalarType, unsigned int length>
  ScalarType vector_norm(Core::LinAlg::Matrix<length, 1, ScalarType> v)
  {
    ScalarType norm_squared = 0.0;
    for (unsigned int i = 0; i < length; i++)
    {
      norm_squared += v(i) * v(i);
    }

    return Core::FADUtils::sqrt<ScalarType>(norm_squared);
  }

  /*!
  \brief Template specialization for double
  */
  template <unsigned int length>
  double vector_norm(Core::LinAlg::Matrix<length, 1, double> v)
  {
    return v.norm2();
  }

  //! Calculates the Norm of a FAD vector, since .Norm2() is not available for FAD vectors
  // Todo this function is obsolete
  template <typename T>
  T norm(Core::LinAlg::Matrix<3, 1, T> v)
  {
    T norm_squared = 0.0;
    for (int i = 0; i < 3; i++)
    {
      norm_squared += v(i) * v(i);
    }

    return Core::FADUtils::sqrt(norm_squared);
  }

  /*!
  \brief Calculate inner product of two FAD or double vectors
  */
  // Todo this function is obsolete, use Dot of Core::LinAlg::Matrix instead
  template <typename Type>
  Type scalar_product(Core::LinAlg::Matrix<3, 1, Type> a, Core::LinAlg::Matrix<3, 1, Type> b)
  {
    return a(0) * b(0) + a(1) * b(1) + a(2) * b(2);
  }

  /*!
  \brief Calculate difference of two FAD or double vectors
  */
  // Todo this function is obsolete, use Update of Core::LinAlg::Matrix instead
  template <typename Type>
  Core::LinAlg::Matrix<3, 1, Type> diff_vector(
      Core::LinAlg::Matrix<3, 1, Type> a, Core::LinAlg::Matrix<3, 1, Type> b)
  {
    Core::LinAlg::Matrix<3, 1, Type> c(Core::LinAlg::Initialization::zero);
    for (int i = 0; i < 3; i++) c(i) = a(i) - b(i);

    return c;
  }

  /*!
  \brief Calculate vector product of two FAD or double vectors
  */
  // Todo this function is obsolete, use cross_product of Core::LinAlg::Matrix instead
  template <typename Type>
  Core::LinAlg::Matrix<3, 1, Type> vector_product(
      Core::LinAlg::Matrix<3, 1, Type> first_vector, Core::LinAlg::Matrix<3, 1, Type> second_vector)
  {
    Core::LinAlg::Matrix<3, 1, Type> result_vector;
    result_vector.clear();
    Core::LinAlg::Matrix<3, 3, Type> S_first_vector;
    S_first_vector.clear();

    S_first_vector(0, 0) = 0.0;
    S_first_vector(0, 1) = -first_vector(2);
    S_first_vector(0, 2) = first_vector(1);
    S_first_vector(1, 0) = first_vector(2);
    S_first_vector(1, 1) = 0.0;
    S_first_vector(1, 2) = -first_vector(0);
    S_first_vector(2, 0) = -first_vector(1);
    S_first_vector(2, 1) = first_vector(0);
    S_first_vector(2, 2) = 0.0;

    result_vector.multiply(S_first_vector, second_vector);

    return result_vector;
  }


  /**
   * \brief Build nested Fad type for computing derivatives up to order N.
   *
   * This function is a slightly modified version of MakeFad from
   * trilinos-source/packages/sacado/example/high_order_example.cpp
   *
   * @tparam N Number of nested derivatives, i.e. order of derivatives to compute.
   * @tparam BaseFadType Basic Fad type of this nested type.
   */
  template <int n, typename BaseFadType>
  struct HigherOrderFadType
  {
    //! Nested type of this Fad type.
    typedef typename HigherOrderFadType<n - 1, BaseFadType>::type nested_type;

    //! Fad type of this object.
    typedef typename Sacado::mpl::apply<BaseFadType, nested_type>::type type;
  };

  /**
   * \brief Specialization for the last nested type, i.e. first order derivatives.
   */
  template <typename BaseFadType>
  struct HigherOrderFadType<1, BaseFadType>
  {
    typedef BaseFadType type;
  };


  /**
   * \brief Create a variable of a nested FAD type.
   *
   * This function is a slightly modified version of MakeFad from
   * trilinos-source/packages/sacado/example/high_order_example.cpp
   *
   * @tparam FadType Fad type of the variable to create.
   */
  template <typename FadType>
  struct HigherOrderFadValue
  {
    //! Type of nested value.
    typedef typename FadType::value_type nested_type;

    /**
     * \brief Set a value of this Fad type.
     * @param n (in) Number of dependent variables.
     * @param i (in) Index of the current dependent variable.
     * @param x (in) Value of the current dependent variable.
     * @return Created Fad type variable.
     */
    static FadType apply(const int n, const int i, const double x)
    {
      return FadType(n, i, HigherOrderFadValue<nested_type>::apply(n, i, x));
    }
  };

  /**
   * \brief Specialization for the last nested type, i.e. double.
   */
  template <>
  struct HigherOrderFadValue<double>
  {
    static double apply(const int n, const int i, const double x) { return x; }
  };

}  // namespace Core::FADUtils

FOUR_C_NAMESPACE_CLOSE

#endif

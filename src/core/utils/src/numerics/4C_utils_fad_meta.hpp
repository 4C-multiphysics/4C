// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FAD_META_HPP
#define FOUR_C_UTILS_FAD_META_HPP

#include "4C_config.hpp"

#include <concepts>
#include <functional>
#include <type_traits>


FOUR_C_NAMESPACE_OPEN

namespace Core::FADUtils
{
  /*!
   * @brief A concept that checks whether a type is a Sacado-Fad type (without actually needing to
   * include the heavy Sacado header).
   *
   * @tparam Scalar
   */
  template <typename Scalar>
  concept SacadoFadType = requires(Scalar a) {
    { a.val() } -> std::convertible_to<typename Scalar::value_type>;
    { a.dx(0) } -> std::convertible_to<typename Scalar::value_type>;
  };

  namespace Internal
  {
    // Note: A Fad-expression has a ExprT1 or a ExprT2 typedef if it is an Expression (and not a
    // pure FAD-Type)
    template <typename T>
    concept HasExpr1Type = requires { typename T::ExprT1; };
    template <typename T>
    concept HasExpr2Type = requires { typename T::ExprT2; };

    /*!
     * @brief A small helper meta-utility to extract the base type of a FAD expression.
     *
     * @note It relies on the the fact that a FAD expression have an ExprT1 or an ExprT2 typedef.
     */
    template <typename Expr>
    struct ExtractAutoDiffBaseType;

    template <typename Expr>
      requires(SacadoFadType<Expr> && !HasExpr1Type<Expr> && !HasExpr2Type<Expr>)
    struct ExtractAutoDiffBaseType<Expr>
    {
      using type = Expr;
    };

    template <typename Expr>
      requires(SacadoFadType<Expr> && HasExpr1Type<Expr>)
    struct ExtractAutoDiffBaseType<Expr>
    {
      using type = ExtractAutoDiffBaseType<typename Expr::ExprT1>::type;
    };

    template <typename Expr>
      requires(SacadoFadType<Expr> && !HasExpr1Type<Expr> && HasExpr2Type<Expr>)
    struct ExtractAutoDiffBaseType<Expr>
    {
      using type = ExtractAutoDiffBaseType<typename Expr::ExprT2>::type;
    };
  }  // namespace Internal

  /*!
   * @brief This template metafunction extracts the base type of a FAD expression from Sacado.
   *
   * Sacado-Fad internally uses lazy evaluation so that the use of auto as a return type is
   * dangerous and fails in certain situations. This class can extract the base type of a FAD
   * expression that can be used as an explicit return type.
   *
   * @note This meta-utilities don't rely on the large Sacado-header.
   *
   * @tparam Expr Arbitrary FAD expression type.
   */
  template <SacadoFadType T>
  using AutoDiffBaseType = typename Internal::ExtractAutoDiffBaseType<T>::type;

  namespace Internal
  {
    template <typename Op>
    struct OperationDeducer;

    template <>
    struct OperationDeducer<std::plus<>>
    {
      template <typename... Ts>
      static constexpr auto apply(const Ts&... a)
      {
        return (a + ...);
      }
    };

    template <>
    struct OperationDeducer<std::minus<>>
    {
      template <typename T1, typename T2>
      static constexpr auto apply(const T1& a, const T2& b)
      {
        return a - b;
      }
    };

    template <>
    struct OperationDeducer<std::multiplies<>>
    {
      template <typename... Ts>
      static constexpr auto apply(const Ts&... a)
      {
        return (a * ...);
      }
    };

    template <>
    struct OperationDeducer<std::divides<>>
    {
      template <typename T1, typename T2>
      static constexpr auto apply(const T1& a, const T2& b)
      {
        return a / b;
      }
    };

    template <typename Operation, typename... Ts>
    struct ScalarOperationResultType;

    template <typename Operation, typename... Ts>
      requires(std::is_arithmetic_v<Ts> && ...)
    struct ScalarOperationResultType<Operation, Ts...>
    {
      using type = decltype(OperationDeducer<Operation>::apply(std::declval<Ts>()...));
    };

    template <typename T1, SacadoFadType T2, typename Operation>
      requires(std::is_arithmetic_v<T1> && std::is_same_v<T1, typename T2::value_type>)
    struct ScalarOperationResultType<Operation, T1, T2>
    {
      using type = AutoDiffBaseType<T2>;
    };

    template <SacadoFadType T1, typename T2, typename Operation>
      requires(std::is_arithmetic_v<T2> && std::is_same_v<T2, typename T1::value_type>)
    struct ScalarOperationResultType<Operation, T1, T2>
    {
      using type = AutoDiffBaseType<T1>;
    };

    template <SacadoFadType T1, SacadoFadType T2, typename Operation>
      requires(std::is_same_v<typename T1::value_type, typename T2::value_type>)
    struct ScalarOperationResultType<Operation, T1, T2>
    {
      using type = AutoDiffBaseType<T1>;
    };
  }  // namespace Internal

  /**
   * @brief A template meta function that determines the result type of a scalar operation that
   * might involves a FAD-type.
   *
   * @note FAD-types are reduced to their base type
   *
   * @tparam Scalar1 The first scalar type involved in the operation.
   * @tparam Scalar2 The second scalar type involved in the operation.
   * @tparam Operation The operation to be applied to the scalar types (std::plus, std::minus,
   * std::multiplies, std::divides).
   */
  template <typename Operation, typename... Scalars>
  using ScalarOperationResultType =
      Internal::ScalarOperationResultType<Operation, Scalars...>::type;
}  // namespace Core::FADUtils

FOUR_C_NAMESPACE_CLOSE

#endif

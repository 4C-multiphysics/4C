// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_SYMBOLIC_EXPRESSION_HPP
#define FOUR_C_UTILS_SYMBOLIC_EXPRESSION_HPP

#include "4C_config.hpp"

#include "4C_utils_symbolic_expression.fwd.hpp"
#include "4C_utils_symbolic_expression_details.hpp"

#include <memory>
#include <string>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /*!
   * @brief A SymbolicExpression evaluates and forms the first and second derivatives of
   * arbitrary symbolic expressions.
   *
   * The class constructs a SymbolicExpression from an expression string. The expression must only
   * contain supported functions ("acos", "asin", "atan", "cos", "sin", "tan", "cosh", "sinh",
   * "tanh", "exp", "log", "log10", "sqrt", "heaviside", "fabs", "atan2") literals
   * ('1.0', 'pi', 'e', 'E', etc) and supported operators ("+", "-", "*", "/", "^", ">", "<", "!",
   * "&&", "||"). In addition, an arbitrary number of variables can be contained. Any substring that
   * is not a number or supported function is parsed as a variable. When calling value(),
   * first_derivative() or second_derivative(), the variables that have been parsed need to be
   * supplied with a value. Note that the expression is parsed and compiled only once, so evaluating
   * the same expression multiple times is efficient.
   *
   * There are two ways to use this class:
   *
   * 1. Compile-time variables: The expression can be defined with compile-time variables, which
   *    are defined as template parameters. This is the preferred, explicit, compile-time checked,
   *    and efficient way to use this class.
   *    Example:
   *
   *    @code
   *    SymbolicExpression<double, "x", "y", "z"> expr("2*x + y/2 - 4*z");
   *    expr.value(var<"x">(1.0), var<"y">(2.0), var<"z">(3.0));
   *    @endcode
   *
   * 2. Run-time variables: The expression can be defined without compile-time variables, in which
   *    case the variables are passed as a map of key-value pairs at runtime. Use this if you have
   *    to, but be aware that this is less efficient and does not provide compile-time checks.
   *    Example:
   *
   *    @code
   *    SymbolicExpression<double> expr("2*x + y/2 - 4*z");
   *    expr.value({{"x", 1.0}, {"y", 2.0}, {"z", 3.0}});
   *    @endcode
   *
   * @note If you want to evaluate the same expression more than once, it is important to reuse a
   * SymbolicExpression instead of creating a new one with the same expression so that the
   * expression only needs to be parsed and compiled once.
   *
   * @tparam Number: Only an arithmetic type is allowed for template parameter. So far only double
   * is supported. In case you use logical operators like ">" or "<" in the expression, you will
   * obtain 1.0 for true and 0.0 for false.
   */

  template <typename Number, CompileTimeString... variables>
  class SymbolicExpression
  {
    static_assert(std::is_arithmetic_v<Number>, "Need an arithmetic type.");

   public:
    //! Type returned by the Value() function
    using ValueType = Number;
    //! Type returned by the FirstDerivative() function
    using FirstDerivativeType = Sacado::Fad::DFad<Number>;
    //! Type returned by the SecondDerivative() function
    using SecondDerivativeType = Sacado::Fad::DFad<Sacado::Fad::DFad<Number>>;

    //! True if the expression contains compile-time variables, false otherwise.
    static constexpr bool has_compile_time_variables = (sizeof...(variables) > 0);

    //! Default constructor without any expression. Assign an expression to this object
    //! before using it.
    SymbolicExpression() = default;

    //! Construct a SymbolicExpression from the given @p expression string. The expression must only
    //! contain supported functions, literals and operators, as well as arbitrary number of
    //! variables. See the class documentation for more details.
    explicit SymbolicExpression(const std::string& expression);

    //! destructor
    ~SymbolicExpression() = default;

    //! copy constructor
    SymbolicExpression(const SymbolicExpression& other);

    //! copy assignment operator
    SymbolicExpression& operator=(const SymbolicExpression& other);

    //! move constructor
    SymbolicExpression(SymbolicExpression&& other) noexcept = default;

    //! move assignment operator
    SymbolicExpression& operator=(SymbolicExpression&& other) noexcept = default;

    /**
     * @brief Evaluates the parsed expression with compile-time variables.
     *
     * Use the var() helper function to create the variables. The variables must be passed in the
     * same order as they were defined in the template parameters.
     */
    template <typename = void>
      requires(has_compile_time_variables)
    ValueType value(VarWrapper<variables, ValueType>&&... vars) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_value_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_value_->evaluate(std::move(vars)...);
    }

    /**
     * @brief Evaluates the parsed expression with runtime variables.
     *
     * Failing to provide a value for a variable that is parsed in the expression will result in an
     * exception being thrown. It is *not* an error to provide a value for a variable that is
     * not parsed in the expression; the variable is simply ignored.
     */
    template <typename = void>
      requires(!has_compile_time_variables)
    ValueType value(const std::map<std::string, ValueType>& args) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_value_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_value_->evaluate(args);
    }


    /**
     * @brief Evaluates the first derivative of the parsed expression with compile-time variables.
     *
     * Use the var() helper function to create the variables. The variables must be passed in the
     * same order as they were defined in the template parameters. The values need to be of
     * the FirstDerivativeType, which means an auto-differentiated type. How you set the values is
     * up to the user. Specifically, it is not necessary to set up every variable for
     * differentiation. Example:
     *
     * @code
     * SymbolicExpression<double, "x", "y", "z"> expr("2*x + y/2 - 4*z");
     * // Differentiating with respect to x and y, but not z.
     * auto diff = expr.first_derivative(var<"x">(FirstDerivativeType(1, 0, 1.0)),
     *   var<"y">(FirstDerivativeType(2, 1, 2.0)),
     *   var<"z">(FirstDerivativeType(3.0)));
     * @endcode
     */
    template <typename = void>
      requires(has_compile_time_variables)
    FirstDerivativeType first_derivative(VarWrapper<variables, FirstDerivativeType>&&... vars) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_first_deriv_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_first_deriv_->evaluate(std::move(vars)...);
    }


    /*!
     * @brief Evaluates the first derivative of the parsed expression with runtime variables.
     *
     * See the other functions in this class for details on how to pass variables and set up the
     * derivative values.
     */
    template <typename = void>
      requires(!has_compile_time_variables)
    FirstDerivativeType first_derivative(
        const std::map<std::string, FirstDerivativeType>& args) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_first_deriv_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_first_deriv_->evaluate(args);
    }


    /*!
     * @brief Evaluates the second derivative of the parsed expression with compile-time variables.
     *
     * See the other functions in this class for details on how to pass variables and set up the
     * derivative values.
     */
    template <typename = void>
      requires(has_compile_time_variables)
    SecondDerivativeType second_derivative(
        VarWrapper<variables, SecondDerivativeType>&&... vars) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_second_deriv_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_second_deriv_->evaluate(std::move(vars)...);
    }


    /**
     * @brief Evaluates the second derivative of the parsed expression with runtime variables.
     *
     * See the other functions in this class for details on how to pass variables and set up the
     * derivative values.
     */
    template <typename = void>
      requires(!has_compile_time_variables)
    SecondDerivativeType second_derivative(
        const std::map<std::string, SecondDerivativeType>& args) const
    {
      FOUR_C_ASSERT_ALWAYS(impl_second_deriv_,
          "SymbolicExpression not initialized. Please assign an expression before using it.");
      return impl_second_deriv_->evaluate(args);
    }

   private:
    std::unique_ptr<Core::Utils::SymbolicExpressionDetails::Impl<ValueType, variables...>>
        impl_value_;

    std::unique_ptr<Core::Utils::SymbolicExpressionDetails::Impl<FirstDerivativeType, variables...>>
        impl_first_deriv_;

    std::unique_ptr<
        Core::Utils::SymbolicExpressionDetails::Impl<SecondDerivativeType, variables...>>
        impl_second_deriv_;
  };


  template <typename Number, Core::Utils::CompileTimeString... variables>
  Core::Utils::SymbolicExpression<Number, variables...>::SymbolicExpression(
      const std::string& expression)
      : impl_value_(
            std::make_unique<Core::Utils::SymbolicExpressionDetails::Impl<ValueType, variables...>>(
                expression)),
        impl_first_deriv_(std::make_unique<
            Core::Utils::SymbolicExpressionDetails::Impl<FirstDerivativeType, variables...>>(
            expression)),
        impl_second_deriv_(std::make_unique<
            Core::Utils::SymbolicExpressionDetails::Impl<SecondDerivativeType, variables...>>(
            expression))
  {
  }



  template <typename Number, Core::Utils::CompileTimeString... variables>
  Core::Utils::SymbolicExpression<Number, variables...>::SymbolicExpression(
      const Core::Utils::SymbolicExpression<Number, variables...>& other)
      : impl_value_{std::make_unique<
            Core::Utils::SymbolicExpressionDetails::Impl<ValueType, variables...>>(
            *other.impl_value_)},
        impl_first_deriv_{std::make_unique<
            Core::Utils::SymbolicExpressionDetails::Impl<FirstDerivativeType, variables...>>(
            *other.impl_first_deriv_)},
        impl_second_deriv_{std::make_unique<
            Core::Utils::SymbolicExpressionDetails::Impl<SecondDerivativeType, variables...>>(
            *other.impl_second_deriv_)}
  {
  }


  template <typename Number, Core::Utils::CompileTimeString... variables>
  Core::Utils::SymbolicExpression<Number, variables...>&
  Core::Utils::SymbolicExpression<Number, variables...>::operator=(
      const Core::Utils::SymbolicExpression<Number, variables...>& other)
  {
    impl_value_ =
        std::make_unique<Core::Utils::SymbolicExpressionDetails::Impl<ValueType, variables...>>(
            *other.impl_value_);
    impl_first_deriv_ = std::make_unique<
        Core::Utils::SymbolicExpressionDetails::Impl<FirstDerivativeType, variables...>>(
        *other.impl_first_deriv_);
    impl_second_deriv_ = std::make_unique<
        Core::Utils::SymbolicExpressionDetails::Impl<SecondDerivativeType, variables...>>(
        *other.impl_second_deriv_);
    return *this;
  }

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

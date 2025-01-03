// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_SYMBOLIC_EXPRESSION_HPP
#define FOUR_C_UTILS_SYMBOLIC_EXPRESSION_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"

#include <Sacado.hpp>

#include <map>
#include <memory>
#include <numeric>
#include <string>

FOUR_C_NAMESPACE_OPEN


namespace Core::Utils
{
  // forward declaration
  namespace SymbolicExpressionDetails
  {
    template <typename Number>
    class Parser;
  }


  /*!
   *  @brief The SymbolicExpression class evaluates and forms the first and second derivatives of
   * arbitrary symbolic expressions.
   *
   * The class constructs a SymbolicExpression from an expression string. The expression must only
   * contain supported functions ("acos", "asin", "atan", "cos", "sin", "tan", "cosh", "sinh",
   * "tanh", "exp", "log", "log10", "sqrt", "heaviside", "fabs", "atan2") literals
   * ('1.0', 'pi', 'e', 'E', etc) and supported operators ("+", "-", "*", "/", "^", ".", ","). In
   * addition, an arbitrary number of variables can be contained. Any substring that is not a number
   * or supported function is parsed as a variable. When calling Value(), FirstDerivative() or
   * SecondDerivative(), the variables that have been parsed need to be supplied with a value.
   *
   * \note If you want to evaluate the same expression more than once, it is better to reuse that
   * object of the SymbolicExpression instead of creating a new object of that class with the same
   * expression so that the expression only needs to be parsed once.
   *
   * @tparam Number: Only an arithmetic type is allowed for template parameter. So far only double
   * is supported.
   */

  template <typename Number>
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

    //! Construct a SymbolicExpression from the given @p expression string. The expression must only
    //! contain supported functions, literals and operators, as well as arbitrary number of
    //! variables. See the class documentation for more details.
    SymbolicExpression(const std::string& expression);

    //! destructor
    ~SymbolicExpression();

    //! copy constructor
    SymbolicExpression(const SymbolicExpression& other);

    //! copy assignment operator
    SymbolicExpression& operator=(const SymbolicExpression& other);

    //! move constructor
    SymbolicExpression(SymbolicExpression&& other) noexcept = default;

    //! move assignment operator
    SymbolicExpression& operator=(SymbolicExpression&& other) noexcept = default;


    /*!
     * @brief evaluates the parsed expression for a given set of variables
     *
     * @param[in] variable_values A map containing all variables (variablename, value) necessary to
     * evaluate the parsed expression. If a parsed variable is not specified, an error is thrown
     * naming the missing variable.
     * @return Value of the parsed expression
     */
    ValueType value(const std::map<std::string, ValueType>& variable_values) const;


    /*!
     * @brief evaluates the first derivative of the parsed expression with respect to a given set of
     * variables. If a parsed variable is not specified in the @p variable_values or @p constants,
     * an error is thrown naming the missing variable.
     *
     * @param[in] variable_values A map containing all variables (variablename, value) necessary to
     * build the first derivative of the parsed expression with respect to this variables. Since the
     * first derivative of the parsed expression is evaluated, only Sacado::Fad::DFad<Number> types,
     * with Number being an arithmetic type, are allowed.
     * @param[in] constants A map containing all constants (constantname, value) necessary
     * to evaluate the parsed expression.
     * @return  First derivative of the parsed expression with respect to the variables
     */
    FirstDerivativeType first_derivative(std::map<std::string, FirstDerivativeType> variable_values,
        const std::map<std::string, ValueType>& constant_values) const;


    /*!
     * @brief evaluates the second derivative of the parsed expression with respect to a given set
     * of variables. If a parsed variable is not specified in the @p variable_values or @p
     * constants, an error is thrown naming the missing variable.
     *
     * @param[in] variable_values A map containing all variables (variablename, value) necessary to
     * build the first derivative of the parsed expression with respect to this variables. Since the
     * second derivative of the parsed expression is evaluated, only
     * Sacado::Fad::DFad<Sacado::Fad::DFad<Number>> types, with Number being an arithmetic type, are
     * allowed.
     * @param[in] constants A map containing all constants (constantname, value) necessary
     * to evaluate the parsed expression.
     * @return  Second derivative of the parsed expression with respect to the variables
     */
    SecondDerivativeType second_derivative(
        const std::map<std::string, SecondDerivativeType>& variable_values,
        const std::map<std::string, ValueType>& constant_values) const;

   private:
    //! Parser for the symbolic expression evaluation
    std::unique_ptr<Core::Utils::SymbolicExpressionDetails::Parser<ValueType>> parser_for_value_;

    //! Parser for the symbolic expression first derivative evaluation
    std::unique_ptr<Core::Utils::SymbolicExpressionDetails::Parser<FirstDerivativeType>>
        parser_for_firstderivative_;

    //! Parser for the symbolic expression second derivative evaluation
    std::unique_ptr<Core::Utils::SymbolicExpressionDetails::Parser<SecondDerivativeType>>
        parser_for_secondderivative_;
  };

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

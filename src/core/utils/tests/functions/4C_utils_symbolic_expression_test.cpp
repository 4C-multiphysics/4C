// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_symbolic_expression.hpp"

#include "4C_unittest_utils_assertions_test.hpp"
#include "4C_utils_exceptions.hpp"
#include "4C_utils_symbolic_expression.fwd.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  using FAD = Sacado::Fad::DFad<double>;
  using Core::Utils::var;

  /// converts the values of variables from type double to FAD double and returns the modified
  /// vector of name-value-pairs
  std::map<std::string, Sacado::Fad::DFad<double>> convert_variable_values_to_fad_objects(
      const std::map<std::string, double>& variables)
  {
    std::map<std::string, Sacado::Fad::DFad<double>> variables_FAD;

    // number of variables
    auto numvariables = static_cast<int>(variables.size());

    // counter for variable numbering
    int counter = 0;

    // set the values of the variables
    for (const auto& [name, value] : variables)
    {
      // FAD object for 1st order derivatives
      Sacado::Fad::DFad<double> varfad(numvariables, counter, value);

      // create name-value-pairs with values now of type FAD double and add to vector
      variables_FAD.emplace(name, varfad);

      // update counter
      counter++;
    }
    return variables_FAD;
  }


  TEST(SymbolicExpressionTest, TestNoVariables)
  {
    Core::Utils::SymbolicExpression<double, "t"> symbolicexpression("2.0");

    EXPECT_DOUBLE_EQ(symbolicexpression.value(Core::Utils::var<"t">(0.)), 2.0);
  }

  TEST(SymbolicExpressionTest, TestValue)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression("2*x");

    EXPECT_DOUBLE_EQ(symbolicexpression.value({{"x", 2.0}}), 4.0);
  }

  TEST(SymbolicExpressionTest, TestFirstDeriv)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression_bilin(
        "2*Variable1*Constant1*Variable2");
    Core::Utils::SymbolicExpression<double> symbolicexpression_xtimesx(
        "2*Variable1*Variable1*Constant1*Variable2*Variable2");
    Core::Utils::SymbolicExpression<double> symbolicexpression_pow2(
        "2*Variable1^2*Constant1*Variable2^2");

    std::map<std::string, double> variables{{"Variable1", 6.0}, {"Variable2", 3.0}};

    auto variable_values = convert_variable_values_to_fad_objects(variables);
    variable_values.emplace("Constant1", 2.0);

    auto fdfad_bilin = symbolicexpression_bilin.first_derivative(variable_values);
    auto fdfad_xtimesx = symbolicexpression_xtimesx.first_derivative(variable_values);
    auto fdfad_pow2 = symbolicexpression_pow2.first_derivative(variable_values);

    EXPECT_DOUBLE_EQ(fdfad_bilin.dx(0), 12.0);                    // dFunction1/dVariable1
    EXPECT_DOUBLE_EQ(fdfad_bilin.dx(1), 24.0);                    // dFunction1/dVariable2
    EXPECT_NEAR(fdfad_xtimesx.dx(0), fdfad_pow2.dx(0), 1.0e-14);  // dFunction2/dVariable1
    EXPECT_NEAR(fdfad_xtimesx.dx(1), fdfad_pow2.dx(1), 1.0e-14);  // dFunction2/dVariable1
  }

  TEST(SymbolicExpressionTest, TestValidFunctionsAndOperators)
  {
    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression_sincostan(
        "2*cos(x) * sin(x) * tan(x) + cosh(x) * sinh(x) * tanh(x) + asin(1.0) * acos(0.5) * "
        "atan(1.0) ");

    Core::Utils::SymbolicExpression<double, "x", "y"> symbolicexpression_logexp(
        " log(exp(1)) * log10(y) - x");

    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression_sqrtheavisidefabs(
        "sqrt(4) + heaviside(3.0) + fabs(2.3) / 1^1");

    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression_atan2("atan2(2,4)");

    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression_xpow2("x^2");
    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression_xtimesx("x * x");

    EXPECT_NEAR(symbolicexpression_sincostan.value(var<"x">(0.2)), 1.4114033869288349, 1.0e-14);

    EXPECT_NEAR(symbolicexpression_logexp.value(var<"x">(0.2), var<"y">(0.4)), -0.59794000867203767,
        1.0e-14);

    EXPECT_NEAR(symbolicexpression_sqrtheavisidefabs.value(var<"x">(1.0)), 5.3, 1.0e-14);

    EXPECT_NEAR(symbolicexpression_atan2.value(var<"x">(1.0)), 0.46364760900080609, 1.0e-14);
    EXPECT_NEAR(symbolicexpression_xpow2.value(var<"x">(0.2)),
        symbolicexpression_xtimesx.value(var<"x">(0.2)), 1.0e-14);
  }

  TEST(SymbolicExpressionTest, TestValidLiterals)
  {
    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression("2*pi * 1.0e-3  + 3.0E-4 * x");

    EXPECT_NEAR(symbolicexpression.value(var<"x">(1.0)), 0.0065831853071795865, 1.0e-14);
  }

  TEST(SymbolicExpressionTest, UnaryMinus)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression("-4.0 * t");

    auto value = symbolicexpression.value({{"t", 1.0}});
    EXPECT_DOUBLE_EQ(value, -4.0);
    auto first_derivative = symbolicexpression.first_derivative({{"t", FAD(1, 0, 1.0)}});
    EXPECT_DOUBLE_EQ(first_derivative.dx(0), -4.0);
  }

  TEST(SymbolicExpressionTest, UselessVariablesDiscarded)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression("1.23");

    auto value = symbolicexpression.value({{"x", 1.0}});
    EXPECT_DOUBLE_EQ(value, 1.23);
  }

  TEST(SymbolicExpressionTest, SingleVariable)
  {
    Core::Utils::SymbolicExpression<double, "x"> symbolicexpression("x");
    auto value = symbolicexpression.value(var<"x">(1.0));
    EXPECT_DOUBLE_EQ(value, 1.0);
  }

  TEST(SymbolicExpressionTest, EvaluateWithMissingVariableThrows)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression(
        "2*Variable1*Constant1*Variable2*Variable3");

    EXPECT_ANY_THROW(symbolicexpression.value({{"Variable1", 1.0}, {"Constant1", 1.0}}));
  }

  TEST(SymbolicExpressionTest, InvalidOperatorThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::Utils::SymbolicExpression<double> symbolicexpression("2 ** 4"), Core::Exception,
        R"(Error while parsing:
2 ** 4
   ^
   Expected a primary expression.)");
  }


  TEST(SymbolicExpressionTest, MissingBracketsThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::Utils::SymbolicExpression<double> symbolicexpression("2*4 - (3 + 1"), Core::Exception,
        R"(Error while parsing:
2*4 - (3 + 1
           ^
           Expected closing parenthesis.)");
  }

  TEST(SymbolicExpressionTest, IncompleteFunctionThrows)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::Utils::SymbolicExpression<double> symbolicexpression("2*4 - (3 + "), Core::Exception,
        R"(Error while parsing:
2*4 - (3 + 
          ^
          Expected a primary expression.)");
  }

  TEST(SymbolicExpressionTest, Copyable)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression("2*x + y + 4*z");
    std::map<std::string, double> variables = {{"x", 1.0}, {"y", 2.0}, {"z", 3.0}};

    auto copy = symbolicexpression;
    EXPECT_DOUBLE_EQ(copy.value(variables), 16.0);

    Core::Utils::SymbolicExpression<double> another_expression("x");
    another_expression = copy;
    EXPECT_DOUBLE_EQ(another_expression.value(variables), 16.0);
  }


  TEST(SymbolicExpressionTest, Moveable)
  {
    Core::Utils::SymbolicExpression<double> symbolicexpression("2*x + y + 4*z");
    std::map<std::string, double> variables = {{"x", 1.0}, {"y", 2.0}, {"z", 3.0}};

    Core::Utils::SymbolicExpression<double> moved_expression(std::move(symbolicexpression));
    EXPECT_DOUBLE_EQ(moved_expression.value(variables), 16.0);

    Core::Utils::SymbolicExpression<double> another_expression("x");
    another_expression = std::move(moved_expression);
    EXPECT_DOUBLE_EQ(another_expression.value(variables), 16.0);
  }

  TEST(SymbolicExpressionTest, CompileTimeStrings)
  {
    constexpr Core::Utils::CompileTimeString x1{"x"};
    constexpr Core::Utils::CompileTimeString x2{"x"};
    constexpr Core::Utils::CompileTimeString y{"y"};
    constexpr Core::Utils::CompileTimeString z{"abc"};

    static_assert(x1 == x2);
    static_assert(x1 != y);
    static_assert(x1 != z);

    static_assert(Core::Utils::index_of<"x", "x">() == 0);
    static_assert(Core::Utils::index_of<"x", "other", "x", "another">() == 1);
    static_assert(Core::Utils::index_of<"x", "other", "another">() == -1);
  }

  TEST(SymbolicExpressionTest, CompileTimeVariables)
  {
    Core::Utils::SymbolicExpression<double, "x", "y", "z"> symbolicexpression("2*x + y + 4*z");
    auto value = symbolicexpression.value(
        Core::Utils::var<"x">(1.0), Core::Utils::var<"y">(2.0), Core::Utils::var<"z">(3.0));
    EXPECT_DOUBLE_EQ(value, 16.0);
  }

  TEST(SymbolicExpressionTest, Comparison)
  {
    Core::Utils::SymbolicExpression<double, "x", "y"> expr("x*y > 0");
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(2.0), Core::Utils::var<"y">(3.0)), 1.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(2.0), Core::Utils::var<"y">(-3.0)), 0.0);
  }

  TEST(SymbolicExpressionTest, Equality)
  {
    Core::Utils::SymbolicExpression<double, "x", "y"> expr("x == 2^2*y");
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(4.0), Core::Utils::var<"y">(1.0)), 1.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(0.0), Core::Utils::var<"y">(-3.0)), 0.0);
  }

  TEST(SymbolicExpressionTest, NotEqual)
  {
    Core::Utils::SymbolicExpression<double, "x", "y"> expr("x != 2^2*y");
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(4.0), Core::Utils::var<"y">(1.0)), 0.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(0.0), Core::Utils::var<"y">(-3.0)), 1.0);
  }

  TEST(SymbolicExpressionTest, SumOfComparisons)
  {
    Core::Utils::SymbolicExpression<double, "x", "y"> expr("(x*y > 0) + (x >= y)");
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(2.0), Core::Utils::var<"y">(1.0)), 2.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(0.0), Core::Utils::var<"y">(0.0)), 1.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(0.0), Core::Utils::var<"y">(1.0)), 0.0);
  }

  TEST(SymbolicExpressionTest, LogicalOperators)
  {
    // Note that || has lower precedence than &&
    Core::Utils::SymbolicExpression<double, "x", "y"> expr(
        "(x > 1.0 - 1.0) && (y > sin(0)) || (x <= !2) && !(y > 0)");
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(2.0), Core::Utils::var<"y">(3.0)), 1.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(-2.0), Core::Utils::var<"y">(-3.0)), 1.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(2.0), Core::Utils::var<"y">(-3.0)), 0.0);
    EXPECT_DOUBLE_EQ(expr.value(Core::Utils::var<"x">(0.0), Core::Utils::var<"y">(3.0)), 0.0);
  }

  TEST(SymbolicExpressionTest, LexerError)
  {
    FOUR_C_EXPECT_THROW_WITH_MESSAGE(
        Core::Utils::SymbolicExpression<double> symbolicexpression("2*x % y"), Core::Exception,
        R"(Error while reading:
2*x % y
    ^
    Unknown character.)");
  }


}  // namespace
FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include <gtest/gtest.h>

#include "4C_utils_functionvariables.hpp"

FOUR_C_NAMESPACE_OPEN

namespace
{
  class FunctionVariableStub : public Core::Utils::FunctionVariable
  {
   public:
    FunctionVariableStub(double start, double end, double returnValue = 0.0)
        : FunctionVariable("stub"), start_(start), end_(end), return_value_(returnValue)
    {
    }

    double value(const double t) override { return return_value_; }

    double time_derivative_value(const double t, const unsigned int deg) override
    {
      return return_value_;
    }

    bool contain_time(const double t) override { return t >= start_ && t <= end_; }

   private:
    const double start_;
    const double end_;
    const double return_value_;
  };

  TEST(PiecewiseVariableTest, ContainTimeSingle)
  {
    auto stub = std::make_shared<FunctionVariableStub>(0.0, 1.0);
    Core::Utils::PiecewiseVariable piecewiseVariable("test", {stub});

    EXPECT_TRUE(piecewiseVariable.contain_time(0.0));
    EXPECT_TRUE(piecewiseVariable.contain_time(0.5));
    EXPECT_TRUE(piecewiseVariable.contain_time(1.0));
    EXPECT_TRUE(not piecewiseVariable.contain_time(2.0));
  }

  TEST(PiecewiseVariableTest, ValueSingle)
  {
    auto stub = std::make_shared<FunctionVariableStub>(0.0, 1.0);
    Core::Utils::PiecewiseVariable piecewiseVariable("test", {stub});

    EXPECT_EQ(piecewiseVariable.value(0.0), 0.0);
    EXPECT_EQ(piecewiseVariable.value(0.5), 0.0);
    EXPECT_EQ(piecewiseVariable.value(1.0), 0.0);
    EXPECT_ANY_THROW(piecewiseVariable.value(2.0));
  }

  TEST(PiecewiseVariableTest, TimeDerivativeValueSingle)
  {
    auto stub = std::make_shared<FunctionVariableStub>(0.0, 1.0);
    Core::Utils::PiecewiseVariable piecewiseVariable("test", {stub});

    EXPECT_EQ(piecewiseVariable.time_derivative_value(0.0, 1), 0.0);
    EXPECT_EQ(piecewiseVariable.time_derivative_value(0.5, 1), 0.0);
    EXPECT_EQ(piecewiseVariable.time_derivative_value(1.0, 1), 0.0);
    EXPECT_ANY_THROW(piecewiseVariable.time_derivative_value(2.0, 1));
  }

  TEST(PiecewiseVariableTest, ContainTimeMultiple)
  {
    // define a piece-wise variable which has overlapping areas
    Core::Utils::PiecewiseVariable piecewiseVariable(
        "test", {std::make_shared<FunctionVariableStub>(0.0, 1.0, 1.0),
                    std::make_shared<FunctionVariableStub>(0.5, 2.0, 2.0),
                    std::make_shared<FunctionVariableStub>(1.0, 3.0, 3.0)});

    EXPECT_TRUE(piecewiseVariable.contain_time(0.0));
    EXPECT_TRUE(piecewiseVariable.contain_time(1.0));
    EXPECT_TRUE(piecewiseVariable.contain_time(2.0));
    EXPECT_TRUE(piecewiseVariable.contain_time(3.0));

    EXPECT_TRUE(not piecewiseVariable.contain_time(4.0));
  }

  TEST(PiecewiseVariableTest, ValueMultiple)
  {
    // define a piece-wise variable which has overlapping areas
    Core::Utils::PiecewiseVariable piecewiseVariable(
        "test", {std::make_shared<FunctionVariableStub>(0.0, 1.0, 1.0),
                    std::make_shared<FunctionVariableStub>(0.5, 2.0, 2.0),
                    std::make_shared<FunctionVariableStub>(1.0, 3.0, 3.0)});

    EXPECT_EQ(piecewiseVariable.value(0.0), 1.0);
    EXPECT_EQ(piecewiseVariable.value(0.8), 1.0);
    EXPECT_EQ(piecewiseVariable.value(1.5), 2.0);
    EXPECT_EQ(piecewiseVariable.value(2.1), 3.0);
  }

  TEST(PiecewiseVariableTest, TimeDerivativeValueMultiple)
  {
    // define a piece-wise variable which has overlapping areas
    Core::Utils::PiecewiseVariable piecewiseVariable(
        "test", {std::make_shared<FunctionVariableStub>(0.0, 1.0, 1.0),
                    std::make_shared<FunctionVariableStub>(0.5, 2.0, 2.0),
                    std::make_shared<FunctionVariableStub>(1.0, 3.0, 3.0)});

    EXPECT_EQ(piecewiseVariable.time_derivative_value(0.0, 0), 1.0);
    EXPECT_EQ(piecewiseVariable.time_derivative_value(0.8, 0), 1.0);
    EXPECT_EQ(piecewiseVariable.time_derivative_value(1.5, 0), 2.0);
    EXPECT_EQ(piecewiseVariable.time_derivative_value(2.1, 0), 3.0);
  }
}  // namespace

FOUR_C_NAMESPACE_CLOSE

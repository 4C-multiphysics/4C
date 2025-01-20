// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FUNCTION_OF_TIME_HPP
#define FOUR_C_UTILS_FUNCTION_OF_TIME_HPP

#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_functionvariables.hpp"

#include <complex>
#include <iostream>
#include <memory>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /*!
   * \brief interface for time-dependent functions.
   *
   * It encodes potentially vector-valued functions \f$ y_i = f_i(t) \f$ which take a time value
   * \f$ t \f$ and return the component \f$ y_i \f$ or its first derivative.

   */
  class FunctionOfTime
  {
   public:
    /**
     * Virtual destructor.
     */
    virtual ~FunctionOfTime() = default;

    /**
     * Evaluate the function for the given @p time and @p component.
     */
    [[nodiscard]] virtual double evaluate(double time, std::size_t component = 0) const = 0;

    /**
     * Evaluate the derivative of the function for the given @p time and @p component.
     */
    [[nodiscard]] virtual double evaluate_derivative(
        double time, std::size_t component = 0) const = 0;
  };

  /**
   * @brief Function based on user-supplied expressions
   *
   * This class supports functions of type \f$ f( t, a_1(t), ..., a_k(t)) \f$, where
   *  \f$ a_1(t), ..., a_k(t) \f$ are time-dependent FunctionVariable objects.
   */
  class SymbolicFunctionOfTime : public FunctionOfTime
  {
   public:
    /**
     * Create a SymbolicFunctionOfTime From a vector of @p expressions and a vector of @p variables.
     * Any time-dependent variables basing on the FunctionVariable must be passed in the @p
     * variables vector.
     */
    SymbolicFunctionOfTime(const std::vector<std::string>& expressions,
        std::vector<std::shared_ptr<FunctionVariable>> variables);

    [[nodiscard]] double evaluate(double time, std::size_t component = 0) const override;

    [[nodiscard]] double evaluate_derivative(double time, std::size_t component = 0) const override;

   private:
    //! vector of parsed expressions
    std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<double>>> expr_;


    //! vector of the function variables and all their definitions
    std::vector<std::shared_ptr<FunctionVariable>> variables_;
  };

  //! create a vector function of time from multiple expressions
  std::shared_ptr<FunctionOfTime> try_create_function_of_time(
      const std::vector<Core::IO::InputParameterContainer>& parameters);

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

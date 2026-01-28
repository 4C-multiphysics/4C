// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FUNCTION_HPP
#define FOUR_C_UTILS_FUNCTION_HPP


#include "4C_config.hpp"

#include "4C_utils_exceptions.hpp"
#include "4C_utils_functionvariables.hpp"
#include "4C_utils_symbolic_expression.fwd.hpp"

#include <memory>
#include <span>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  /*!
   * \brief interface for space- and time-dependent functions.
   *
   * Functions can be defined either scalar-valued or vector-valued.
   */
  class FunctionOfSpaceTime
  {
   public:
    //! Virtual destructor.
    virtual ~FunctionOfSpaceTime() = default;

    /*!
     * @brief Evaluation of time and space dependent function
     *
     * Evaluate the specified component of the function at the specified position and point in
     * time.
     *
     * @param x  (i) The point in 3-dimensional space in which the function will be evaluated
     * @param t  (i) The point in time in which the function will be evaluated
     * @param component (i) For vector-valued functions, index defines the function-component
     *                      which should be evaluated
     * @return function value
     */
    virtual double evaluate(std::span<const double> x, double t, std::size_t component) const = 0;

    /*!
     * @brief Vector-valued evaluation of time and space dependent function
     *
     * Evaluate the function at the specified position and point in time and return the vector of
     * all components.
     *
     * @param x the point in 3-dimensional space in which the function will be evaluated
     * @param t the point in time in which the function will be evaluated
     * @param values function values
     */
    virtual void evaluate_vector(
        std::span<const double> x, double t, std::span<double> values) const;


    /*!
     * \brief Evaluation of first spatial derivative of time and space dependent function
     *
     * \param x  (i) The point in 3-dimensional space in which the function will be evaluated
     * \param t  (i) The point in time in which the function will be evaluated
     * @param component (i) For vector-valued functions, index defines the function-component
     *                      which should be evaluated
     * \return first spatial derivative of function
     */
    virtual std::vector<double> evaluate_spatial_derivative(
        std::span<const double> x, double t, std::size_t component) const;

    /*!
     * \brief Evaluation of time derivatives and value of the time and space dependent function
     *
     * Evaluate the specified component of the function at the specified position and point in
     * time and calculate the time derivative(s) up to degree @p deg.
     *
     * @param x  (i) The point in 3-dimensional space in which the function will be evaluated
     * @param t  (i) The point in time in which the function will be evaluated
     * @param deg   (i) maximum time derivative degree
     * @param component (i) For vector-valued functions, index defines the function-component
     *                      which should be evaluated
     * @return vector containing value and time derivative(s)
     */
    virtual std::vector<double> evaluate_time_derivative(
        std::span<const double> x, double t, unsigned deg, std::size_t component) const;

    /// Return number of components of function
    [[nodiscard]] virtual std::size_t number_components() const = 0;
  };


  /**
   * @brief Function based on user-supplied expressions
   *
   * This class supports functions of type \f$ f(\mathbf{x}, t, a_1(t), ..., a_k(t)) \f$, where
   * \f$ \mathbf{x} \f$ is the spatial coordinate with `dim` components and where \f$ a_1(t), ...,
   * a_k(t) \f$ are time-dependent FunctionVariable objects.
   */
  class SymbolicFunctionOfSpaceTime : public FunctionOfSpaceTime
  {
   public:
    /**
     * Create an SymbolicFunctionOfSpaceTime. Each entry in @p expressions corresponds to one
     * component of the function, thus the resulting function will have `expressions.size()`
     * components. Any time-dependent variables that appear in the expressions must be passed in
     * the @p variables vector.
     */
    SymbolicFunctionOfSpaceTime(const std::vector<std::string>& expressions,
        std::vector<std::shared_ptr<FunctionVariable>> variables);

    double evaluate(std::span<const double> x, double t, std::size_t component) const override;

    std::vector<double> evaluate_spatial_derivative(
        std::span<const double> x, double t, std::size_t component) const override;

    std::vector<double> evaluate_time_derivative(
        std::span<const double> x, double t, unsigned deg, std::size_t component) const override;

    [[nodiscard]] std::size_t number_components() const override { return (expr_.size()); }

   private:
    /// vector of parsed expressions
    std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<double>>> expr_;

    /// vector of the function variables and all their definitions
    std::vector<std::shared_ptr<FunctionVariable>> variables_;
  };


  /*!
   * \brief Interface for mathematical functions and with arbitrary arguments
   *
   * Functions that derive from this interface are free to take arbitrary arguments for
   * evaluation. It is rather obvious that such an interface would encompass all specialized
   * interfaces for functions. Indeed, any function could be implemented under this interface.
   * However, a lot of our functions have a clear interface, e.g. a function depending only on
   * space and time. There are specialized interfaces for those functions and the present
   * interface should not be misused for these cases. This interface is best used for convenience
   * when prototyping new equations. If a new function type emerges during this process it is
   * likely a good idea to create a new interface for it. Since evaluation of a FunctionOfAnything
   * must always go through another level of indirection due to the data structure for the passed
   * variables, there are two pitfalls: one can either pass variables (from the outside) that are
   * not needed or try to access variables that are not passed (inside the function). Derived
   * classes need to deal with this problem on their own. An important derived class of this
   * interface is the SymbolicFunctionOfAnything.
   *
   * This class evaluates and forms the derivative of arbitrary functions for a given set of
   * variables and constants.
   *
   * The functions can be defined either scalar-valued or vector-valued.
   *
   */
  class FunctionOfAnything
  {
   public:
    //! Virtual destructor.
    virtual ~FunctionOfAnything() = default;

    /*!
     * \brief evaluate function for a given set of variables and constants
     *
     * \note there is no distinction between the input arguments variables and constants. So for
     * the function evaluation it makes no difference whether all necessary variables and
     * constants are passed in a single vector together with an empty vector or the variables and
     * constants are passed separately in individual vectors.
     *
     * \param variables (i) A vector containing a pair (variablename, value) for
     * each variable
     * \param constants (i) A vector containing a pair (variablename, value) for each
     * constant
     * \param component (i) For vector-valued functions, component defines the function-component
     * which should be evaluated
     */
    virtual double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const = 0;

    /*!
     * \brief evaluates the derivative of a function with respect to the given variables
     *
     * \param variables (i) A vector containing a pair (variablename, value) for
     * each variable. The derivative of the function is build with respect to all these variables.
     * \param constants (i) A vector containing a pair (variablename, value) for each
     * constant.
     * \param component (i) For vector-valued functions, component defines the function-component
     * which should be evaluated
     */
    virtual std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const = 0;

    //! Return number of components of function
    [[nodiscard]] virtual std::size_t number_components() const = 0;
  };


  /**
   * @brief Function to evaluate and form the derivative of user defined
   * symbolic expressions.
   *
   * The expression must only contain supported functions, literals and operators, as well as
   * arbitrary number of variables and constants. See the class documentation of
   * SymbolicExpression for more details.
   *
   * It is possible to predefine values of constants in the input file.
   */
  class SymbolicFunctionOfAnything : public FunctionOfAnything
  {
   public:
    SymbolicFunctionOfAnything(
        const std::string& component, std::vector<std::pair<std::string, double>> constants);


    double evaluate(const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;


    std::vector<double> evaluate_derivative(
        const std::vector<std::pair<std::string, double>>& variables,
        const std::vector<std::pair<std::string, double>>& constants,
        const std::size_t component) const override;

    /// return the number of components
    [[nodiscard]] std::size_t number_components() const override { return (expr_.size()); }

   private:
    using ValueType = double;

    //! vector of parsed expressions
    std::vector<std::shared_ptr<Core::Utils::SymbolicExpression<ValueType>>> expr_;

    //! vector of the function variables and all their definitions
    std::vector<std::vector<std::shared_ptr<FunctionVariable>>> variables_;

   private:
    //! constants from input
    std::vector<std::pair<std::string, ValueType>> constants_from_input_;
  };

  /// try to create SymbolicFunctionOfAnything from a given line definition
  std::shared_ptr<FunctionOfAnything> try_create_symbolic_function_of_anything(
      const std::vector<Core::IO::InputParameterContainer>& parameters);

  /// create a vector function from multiple expressions
  std::shared_ptr<FunctionOfSpaceTime> try_create_symbolic_function_of_space_time(
      const std::vector<Core::IO::InputParameterContainer>& parameters);
}  // namespace Core::Utils



FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_FUNCTION_LIBRARY_HPP
#define FOUR_C_UTILS_FUNCTION_LIBRARY_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_utils_function_of_scalar.hpp"

#include <memory>
#include <string>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  class CubicSplineInterpolation;
  class FunctionManager;
}  // namespace Core::Utils

namespace Core::Utils
{
  /// add valid function lines
  void add_valid_library_functions(Core::Utils::FunctionManager& function_manager);

  /**
   * @brief special implementation of a 1D polynomial function
   *
   * This class defines a generic polynomial that that takes one argument. The polynomial can be
   * evaluated and the first derivative is also provided.
   *
   * This class does circumvent using automatic differentiation and may be used in
   * performance-sensitive applications (e.g. operations on Gauss-point level).
   * In particular, evaluating the derivative is faster for this function compared to
   * a SymbolicFunctionOfAnything representing the same polynomial.
   */
  class FastPolynomialFunction : public FunctionOfScalar
  {
   public:
    /**
     * ctor
     *
     * @param coefficients the coefficients of the monomials in ascending order
     */
    FastPolynomialFunction(std::vector<double> coefficients);

    /**
     * @brief Evaluate the polynomial.
     *
     * This is the only supported evaluation call for this class
     *
     * @param argument point to evaluate
     * @return value of the polynomial at @argument
     */
    [[nodiscard]] double evaluate(const double argument) const override;

    /**
     * Evaluate the @deriv_order derivative of polynomial.
     *
     * @param argument point to evaluate
     * @return value of @deriv_order derivative of polynomial at @argument
     */
    [[nodiscard]] double evaluate_derivative(double argument, int deriv_order) const override;

   private:
    //! internal polynomial representation
    const Core::FE::Polynomial mypoly_;
  };

  /**
   * @brief Cubic spline interpolation from csv file
   *
   * This class defines a cubic spline interpolation connecting \f$ n \f$ points
   * \f$ (x_i,y_i), i \in [0,1,...,n-1] \f$ expressed as splines \f$S_i(x)\f$ of the form:
   * \f[
   * S_i(x) = a_i + b_i(x-x_i) + c_i(x-x_i)^2 + d_i(x-x_i)^3, j \in [0,1,...,n-1],
   * \f]
   * where \f$ S_i(x) \f$ is a third order polynomial on the interval \f$ [x_i, x_{i+1}] \f$.
   *
   * @note The spline is calculated such that the first and second derivatives match at the
   * intersection of two segments. Furthermore, so-called natural boundary conditions, i.e. the
   * second derivatives at the two boundaries are set to zero, are applied.
   */
  class CubicSplineFromCSV : public FunctionOfScalar
  {
   public:
    /*!
     * @brief Constructor of function defining a cubic spline interpolation created based on data
     * from csv-file
     *
     * @param[in] csv_file  absolute path to csv file
     */
    CubicSplineFromCSV(const std::string& csv_file);

    [[nodiscard]] double evaluate(double scalar) const override;

    [[nodiscard]] double evaluate_derivative(double scalar, int deriv_order) const override;

   private:
    std::unique_ptr<Core::Utils::CubicSplineInterpolation> cubic_spline_;
  };
}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

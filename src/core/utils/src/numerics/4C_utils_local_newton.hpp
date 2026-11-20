// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_UTILS_LOCAL_NEWTON_HPP
#define FOUR_C_UTILS_LOCAL_NEWTON_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_fixedsizematrix_solver.hpp"
#include "4C_linalg_serialdensematrix.hpp"
#include "4C_linalg_serialdensesolver.hpp"
#include "4C_linalg_serialdensevector.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_utils_densematrix_inverse.hpp"
#include "4C_utils_fad.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

FOUR_C_NAMESPACE_OPEN

namespace Core::Utils
{
  constexpr double LOCAL_NEWTON_DEFAULT_TOLERANCE = 1e-12;
  constexpr unsigned LOCAL_NEWTON_DEFAULT_MAXIMUM_ITERATIONS = 50;

  /// Result of a local Newton solve that may end without convergence.
  template <typename VectorType, typename JacobianType>
  struct LocalNewtonResult
  {
    /// Final Newton iterate. If \c is_converged is true, this is the converged solution. Otherwise,
    /// this is the last computed iterate.
    VectorType x;

    /// Jacobian evaluated at x.
    JacobianType jacobian;

    /// L2 norm of the residual evaluated at x.
    double residuum_norm;

    /// Number of Newton iterations performed.
    unsigned iteration_count;

    /// Whether the Newton method converged within the maximum number of iterations.
    bool is_converged;
  };

  /// @brief Free functions defining a Newton iterations for different scalar, vector and jacobian
  /// types.
  ///
  /// @note These functions are overloaded for common types. If you need to overload this function
  /// for your own types, you may do so in the same namespace that also contains your type.
  /// @{
  template <typename ScalarType>
  void local_newton_iteration(ScalarType& x, const ScalarType residuum, const ScalarType jacobian)
  {
    x -= residuum / jacobian;
  }

  template <unsigned n, typename ScalarType>
  void local_newton_iteration(Core::LinAlg::Matrix<n, 1, ScalarType>& x,
      const Core::LinAlg::Matrix<n, 1, ScalarType>& residuum,
      Core::LinAlg::Matrix<n, n, ScalarType>&& jacobian)
  {
    if constexpr (n < 4)
    {
      Core::LinAlg::invert(jacobian);
    }
    else
    {
      Core::LinAlg::FixedSizeSerialDenseSolver<n, n> serial_dense_solver;
      serial_dense_solver.set_matrix(jacobian);
      serial_dense_solver.invert();
    }
    x.multiply_nn(-1, jacobian, residuum, 1.0);
  }

  template <typename Tensor1, typename Tensor2, typename Tensor3>
    requires(Core::LinAlg::is_tensor<Tensor1> && Core::LinAlg::is_tensor<Tensor2> &&
             Core::LinAlg::is_tensor<Tensor3>)
  void local_newton_iteration(Tensor1& x, const Tensor2& residuum, Tensor3&& jacobian)
  {
    x -= Core::LinAlg::inv(jacobian) * residuum;
  }

  inline void local_newton_iteration(Core::LinAlg::SerialDenseVector& x,
      const Core::LinAlg::SerialDenseVector& residuum, Core::LinAlg::SerialDenseMatrix&& jacobian)
  {
    Core::LinAlg::SerialDenseVector dx(x.length(), true);
    Core::LinAlg::SerialDenseVector rhs(residuum);
    Core::LinAlg::SerialDenseSolver solver;
    solver.set_matrix(jacobian);
    solver.set_vectors(dx, rhs);
    solver.factor_with_equilibration(true);
    solver.factor();
    solver.solve();

    for (int i = 0; i < x.length(); ++i)
    {
      x(i) -= dx(i);
    }
  }
  /// @}

  /// @brief Free functions defining a to compute the L2-norm of the used Vector Type
  ///
  /// @note These functions are overloaded for common types. If you need to overload this function
  /// for your own types, you may do so in the same namespace that also contains your type.
  /// @{
  template <typename DefaultAndFADScalarType>
  DefaultAndFADScalarType l2_norm(const DefaultAndFADScalarType& x)
  {
    return Core::FADUtils::norm(x);
  }

  template <unsigned n, typename ScalarType>
  ScalarType l2_norm(const Core::LinAlg::Matrix<n, 1, ScalarType>& x)
  {
    return Core::FADUtils::vector_norm(x);
  }

  template <std::size_t n, typename ScalarType>
  ScalarType l2_norm(const Core::LinAlg::Tensor<ScalarType, n>& x)
  {
    return Core::LinAlg::norm2(x);
  }

  inline double l2_norm(const Core::LinAlg::SerialDenseVector& x) { return Core::LinAlg::norm2(x); }
  /// @}

  /*!
   * @brief Runs the local Newton method and returns a LocalNewtonResult.
   *
   * @tparam ScalarType The type of the scalar used within the method (type of tolerance / type of
   * residuum norm).
   * @tparam VectorType The type of the residuum and the unknowns.
   * @tparam ResiduumAndJacobianEvaluator A class that defines the operator() with the signature
   * std::tuple<VectorType, JacobianType>(VectorType) that evaluates the residuum and its
   * jacobian at a specific point.
   * @param residuum_and_jacobian_evaluator A function object that evaluates the residuum and its
   * jacobian at a specific point.
   * @param x_0 Initial guess for the solution
   * @param tolerance The tolerance that is used for a convergence criterion.
   * @param max_iterations Maximum allowed number of newton iterations
   * @return The final Newton state. If the method did not converge within the maximum number of
   * iterations, this is reported via LocalNewtonResult::is_converged instead of throwing.
   */
  template <typename ScalarType, typename VectorType, typename ResiduumAndJacobianEvaluator>
  auto solve_local_newton_with_status(ResiduumAndJacobianEvaluator residuum_and_jacobian_evaluator,
      VectorType x_0, const ScalarType tolerance = LOCAL_NEWTON_DEFAULT_TOLERANCE,
      const unsigned max_iterations = LOCAL_NEWTON_DEFAULT_MAXIMUM_ITERATIONS)
      -> LocalNewtonResult<VectorType,
          std::tuple_element_t<1, decltype(residuum_and_jacobian_evaluator(x_0))>>
  {
    auto [residuum, jacobian] = residuum_and_jacobian_evaluator(x_0);
    auto residuum_norm = l2_norm(residuum);

    unsigned iteration = 0;
    while (residuum_norm > tolerance && iteration < max_iterations)
    {
      local_newton_iteration(x_0, residuum, std::move(jacobian));

      std::tie(residuum, jacobian) = residuum_and_jacobian_evaluator(x_0);
      residuum_norm = l2_norm(residuum);

      ++iteration;
    }

    const bool is_converged = (residuum_norm <= tolerance);
    return {.x = std::move(x_0),
        .jacobian = std::move(jacobian),
        .residuum_norm = FADUtils::cast_to_double(residuum_norm),
        .iteration_count = iteration,
        .is_converged = is_converged};
  }

  /*!
   * @brief Finds the root of a function (scalar or vector valued) using the Newton-Raphson method
   * starting from the initial guess @p x_0.
   *
   * @note In order that this function works well, you may need to overload the functions
   * `ScalarType L2Norm(const VectorType&)` that computes the L2-norm of the used vector type, and
   * `void local_newton_iteration(VectorType& x, const VectorType& residuum, JacobianType&&
   * jacobian_of_residuum)` that does a local Newton step. For often used parameters (double,
   * Core::LinAlg::Matrix and FAD-types), these overloads are already implemented.
   *
   * @note The jacobian at the root is often needed to compute the linearization of the
   * Newton-Raphson method w.r.t the primary variables. @p solve_local_newton_and_return_jacobian
   * returns the derivative of the residuum w.r.t. the unknown parameter x, i.e. \f$\frac{\partial
   * \boldsymbol{R}}{\partial \boldsymbol{x}}\f$. For the linearization of your method, you usually
   * need the derivative of x w.r.t. the primary variable u. This can be computed via
   * \f$\frac{\partial \boldsymbol{x}}{\partial \boldsymbol{u}} = (\frac{\partial
   * \boldsymbol{R}}{\partial \boldsymbol{x}})^{-1} \frac{\partial \boldsymbol{R}}{\partial
   * \boldsymbol{u}}\f$, where \f$\frac{\partial \boldsymbol{R}}{\partial \boldsymbol{u}}\f$ is the
   * partial derivative of the residuum R w.r.t. the primary variable u. If you don't need to
   * linearize your function, you can use @p solve_local_newton which does not return the
   * linearization.
   *
   * You can use the function as follows:
   *
   * Example 1 (scalar-valued double function):
   * @code
   * double x_0 = 0.0;
   * auto [x, jacobian] = Core::Utils::solve_local_newton_and_return_jacobian(x_0, [](double x) {
   *     return std::make_tuple<double, double>({std::pow(x, 2), 2*x});
   *   }, 1e-9);
   * @endcode
   *
   * Example 2 (vector-valued double function):
   * @code
   * Core::LinAlg::Matrix<2,1> x_0(true);
   * auto [x, jacobian] = Core::Utils::solve_local_newton_and_return_jacobian(x_0,
   * [](Core::LinAlg::Matrix<2,1> x) { return std::make_tuple<Core::LinAlg::Matrix<2,1>,
   * Core::LinAlg::Matrix<2,2>>({ Core::LinAlg::Matrix<2,1>{true}, // define your function here
   *       Core::LinAlg::Matrix<2,2>{true} // define your jacobian here
   *     });
   *   }, 1e-9);
   * @endcode
   *
   * Example 3 (Custom type double function):
   * @code
   * namespace {
   *   void local_newton_iteration(MyVectorType& x, MyVectorType residuum, MyJacobianType
   * jacobian)
   *   {
   *     // define your Newton update here
   *   }
   *
   *   double L2Norm(const MyVectorType& x)
   *   {
   *      // define norm of vector here
   *   }
   * }
   *
   * double x_0 = MyVectorType{...}; // initial value
   * auto [x, jacobian] = Core::Utils::solve_local_newton_and_return_jacobian(x_0, [](MyVectorType
   * x) { return std::make_tuple<MyVectorType, MyJacobianType>({ MyVectorType{...}, // define your
   * function here MyJacobianType{...} // define your jacobian here
   *     });
   *   }, 1e-9);
   * @endcode
   *
   * @tparam ScalarType The type of the scalar used within the method (type of tolerance / type of
   * residuum norm).
   * @tparam VectorType The type of the residuum and the unknowns.
   * @tparam ResiduumAndJacobianEvaluator A class that defines the operator() with the signature
   * std::tuple<VectorType, JacobianType>(VectorType) that evaluates the residuum and its
   * jacobian at a specific point.
   * @param residuum_and_jacobian_evaluator A function object that evaluates the residuum and its
   * jacobian at a specific point.
   * @param x_0 Initial guess for the solution
   * @param tolerance The tolerance that is used for a convergence criterion.
   * @param max_iterations Maximum allowed number of newton iterations
   * @return Returns x such that the residuum is smaller than the given tolerance. A pair containing
   * the final result and the jacobian. The jacobian is often needed to compute the linearization of
   * the local Newton method.
   */
  template <typename ScalarType, typename VectorType, typename ResiduumAndJacobianEvaluator>
  auto solve_local_newton_and_return_jacobian(
      ResiduumAndJacobianEvaluator residuum_and_jacobian_evaluator, VectorType x_0,
      const ScalarType tolerance = LOCAL_NEWTON_DEFAULT_TOLERANCE,
      const unsigned max_iterations = LOCAL_NEWTON_DEFAULT_MAXIMUM_ITERATIONS)
      -> std::tuple<VectorType,
          std::tuple_element_t<1, decltype(residuum_and_jacobian_evaluator(x_0))>>
  {
    auto result = solve_local_newton_with_status(
        residuum_and_jacobian_evaluator, x_0, tolerance, max_iterations);

    if (!result.is_converged)
    {
      FOUR_C_THROW(
          "The local Newton method did not converge within {} iterations. Residuum is {:.3e} > "
          "{:.3e}.",
          max_iterations, FADUtils::cast_to_double(result.residuum_norm),
          FADUtils::cast_to_double(tolerance));
    }

    return {std::move(result.x), std::move(result.jacobian)};
  }

  /*!
   * @brief Finds the root of a function (scalar or vector valued) using the Newton-Raphson method
   * starting from the initial guess @p x_0.
   *
   * @note in contrast to @p solve_local_newton_and_return_jacobian, this function does not return
   * the jacobian at the root of the function. The remaining syntax is identical.
   *
   * @tparam ScalarType The type of the scalar used within the method (type of tolerance / type of
   * residuum norm).
   * @tparam VectorType The type of the residuum and the unknowns.
   * @tparam ResiduumAndJacobianEvaluator A class that defines the operator() with the signature
   * std::tuple<VectorType, JacobianType>(VectorType) that evaluates the residuum and its
   * jacobian at a specific point.
   * @param residuum_and_jacobian_evaluator A function object that evaluates the residuum and its
   * jacobian at a specific point.
   * @param x_0 Initial guess for the solution
   * @param tolerance The tolerance that is used for a convergence criterion.
   * @param max_iterations Maximum allowed number of newton iterations
   * @return Returns x such that the residuum is smaller than the given tolerance.
   */
  template <typename ScalarType, typename VectorType, typename ResiduumAndJacobianEvaluator>
  auto solve_local_newton(ResiduumAndJacobianEvaluator residuum_and_jacobian_evaluator,
      VectorType x_0, const ScalarType tolerance = LOCAL_NEWTON_DEFAULT_TOLERANCE,
      const unsigned max_iterations = LOCAL_NEWTON_DEFAULT_MAXIMUM_ITERATIONS) -> VectorType
  {
    return std::get<0>(solve_local_newton_and_return_jacobian(
        residuum_and_jacobian_evaluator, x_0, tolerance, max_iterations));
  }

}  // namespace Core::Utils

FOUR_C_NAMESPACE_CLOSE

#endif

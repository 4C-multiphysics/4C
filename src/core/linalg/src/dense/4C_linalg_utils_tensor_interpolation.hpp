// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_TENSOR_INTERPOLATION_HPP
#define FOUR_C_LINALG_UTILS_TENSOR_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_fem_general_utils_polynomial.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_symmetric_tensor_eigen.hpp"
#include "4C_linalg_tensor.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_exceptions.hpp"


FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// enum class for the error types of the tensor interpolator
  enum class TensorInterpolationErrorType
  {
    NoErrors,             ///< no evaluation errors
    LinSolveFailQMatrix,  ///< the solution of the linear system of equations for the rotation
                          ///< matrix Q failed
    LinSolveFailRMatrix,  ///< the solution of the linear system of equations for the rotation
                          ///< matrix R failed
  };

  /// enum class for the interpolation type of the relative rotation matrices
  enum class RotationInterpolationType
  {
    RotationVector,  ///< interpolation of relative rotation vectors,
    Quaternion,      ///< interpolation of relative quaternions,
  };

  /// enum class for the interpolation type of the eigenvalue matrix
  enum class EigenvalInterpolationType
  {
    LOG,     ///< logarithmic weighted average,
    MLS,     ///< moving least squares,
    LOGMLS,  ///< logarithmic moving least squares,
  };

  /// make error message: error types to error message for the tensor interpolator
  inline std::string make_error_message(const TensorInterpolationErrorType err_type)
  {
    switch (err_type)
    {
      case TensorInterpolationErrorType::NoErrors:
        return "No Errors";
      case TensorInterpolationErrorType::LinSolveFailQMatrix:
        return "The solution of the linear system of equations for the rotation matrix Q "
               "(second-order tensor interpolation) has failed!";
      case TensorInterpolationErrorType::LinSolveFailRMatrix:
        return "The solution of the linear system of equations for the rotation matrix R "
               "(second-order tensor interpolation) has failed!";
      default:
        FOUR_C_THROW("to_string(Core::LinAlg::TensorInterpErrorType): you should not be here!");
    }
  }

  /*!
   * @brief Interpolate a symmetric positive-definite second-order tensor using given weights and
   * tensors according to the R-LOG method.
   *
   * The eigenvalues of the stretch tensor are interpolated using the logarithmic weighted average.
   * The corresponding eigenvectors are interpolated using rotation vector interpolation.
   *
   * @param weights normalized weights
   * @param tensors reference symmetric and positive definite tensors to be interpolated
   * @return Core::LinAlg::SymmetricTensor<double, 3, 3> interpolated symmetric positive definite
   * tensor
   */
  Core::LinAlg::SymmetricTensor<double, 3, 3> interpolate_spd(
      const std::ranges::sized_range auto& weights, const std::ranges::sized_range auto& tensors);

  /*!
   * @brief A default-constructible interpolator type that can be used to interpolate symmetric
   * positive-definite tensors (e.g., in an interpolated input field)
   *
   * @tparam T
   */
  template <typename T>
  struct SymmetricPositiveDefiniteInterpolation
  {
    Core::LinAlg::SymmetricTensor<T, 3, 3> operator()(
        const std::ranges::sized_range auto& weights, const std::ranges::sized_range auto& tensors)
    {
      return interpolate_spd(weights, tensors);
    }
  };

  /*!
   * @brief Interpolate rotation tensors using normalized weights and rotation vector
   * interpolation.
   *
   * @param weights normalized weights
   * @param rotation_tensors reference rotation tensors used for interpolation
   * @return Core::LinAlg::Tensor<double, 3, 3> interpolated rotation tensor
   */
  Core::LinAlg::Tensor<double, 3, 3> interpolate_rotation_tensor(
      const std::ranges::sized_range auto& weights,
      const std::ranges::sized_range auto& rotation_tensors);

  /*!
   * @brief Interpolate the eigenvalues using the logarithmic weighted average method.
   *
   * @note The eigenvalues of the input need to be sorted in a consistent order.
   *
   * @param weights normalized weights
   * @param eigenvalues consistently sorted eigenvalues of the reference tensors
   * @return std::array<double, 3> interpolated eigenvalues
   */
  std::array<double, 3> log_interpolate_eigenvalues(const std::ranges::sized_range auto& weights,
      const std::ranges::sized_range auto& eigenvalues);

  /*!
   * \class SecondOrderTensorInterpolator
   *
   * Interpolation of invertible second-order tensors (3x3), preserving tensor
   * characteristics.
   *
   * The class provides the capability to interpolate a second-order tensor \f$
   * \boldsymbol{T}_{\text{p}}
   * \f$ at the specified location  \f$ \boldsymbol{x}_\text{p} \f$, given a set of tensors \f$
   * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions \f$ \boldsymbol{x}_j \f$.
   * The interpolation scheme, using a combined polar and spectral decomposition, preserves
   * several tensor characteristics, such as positive definiteness and monotonicity of
   * invariants. For further information on the interpolation scheme, refer to:
   * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373
   *
   * @tparam loc_dim dimension of the location vectors \f$ \boldsymbol{x}_j \f$
   */
  template <unsigned int loc_dim>
  class SecondOrderTensorInterpolator
  {
   public:
    /*! @brief Constructor of the second-order tensor interpolator class
     *
     *  @param[in] order polynomial order (1:linear, 2: quadratic, ...) used for interpolating
     * the rotation vectors at the specified location
     *  @param[in] rot_interp_type interpolation algorithm used for
     *  the rotation matrices
     *  @param[in] eigenval_interp_type interpolation algorithm used for
     *  the eigenvalue matrices
     *  @param[in] interp_params interpolation parameters
     */
    SecondOrderTensorInterpolator(unsigned int order,
        const RotationInterpolationType rot_interp_type,
        const EigenvalInterpolationType eigenval_interp_type,
        const ScalarInterpolationParams& interp_params);

    /*! @brief Helper function to define the polynomial space
     *
     *  @param[in] order polynomial order (1:linear, 2: quadratic, ...) used for interpolating
     * the rotation vectors at the specified location
     *  @returns polynomial space(monomials) with desired polynomial order and dimensionality
     */
    Core::FE::PolynomialSpaceComplete<loc_dim, Core::FE::Polynomial> create_polynomial_space(
        unsigned int order);

    /*!
     * @brief Interpolate matrix (second-order 3x3 tensor) from a set of defined reference
     * matrices at specified locations.
     *
     * This method performs tensor interpolation based on a given set of tensors \f$
     * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions/locations \f$
     * \boldsymbol{x}_j \f$. Concretely, the tensor is interpolated at the specified location \f$
     * \boldsymbol{x}_{\text{p}} \f$. Specifically, the R-LOG method from the paper below is
     * currently implemented (rotation vector interpolation + logarithmic weighted average method
     * for eigenvalues):
     * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
     * Second-Order Tensors, Int J Number Methods Eng. 2024, 125, 10.1002/nme.7373
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @returns interpolated 3x3 matrix
     */
    Core::LinAlg::Matrix<3, 3> get_interpolated_matrix(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, TensorInterpolationErrorType& err_type);

    /*!
     * @brief Interpolate matrix (second-order 3x3 tensor) from a set of defined reference
     * matrices at specified locations.
     *
     * This method performs tensor interpolation based on a given set of tensors \f$
     * \boldsymbol{T}_j \f$ (second-order, 3x3) at the spatial positions/locations \f$
     * \boldsymbol{x}_j \f$. Concretely, the tensor is interpolated at the specified location \f$
     * \boldsymbol{x}_{\text{p}} \f$. Specifically, the R-LOG method from the paper below is
     * currently implemented (rotation vector interpolation + logarithmic weighted average method
     * for eigenvalues):
     * -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
     * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @returns interpolated 3x3 matrix
     */
    Core::LinAlg::Matrix<3, 3> get_interpolated_matrix(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<double>& ref_locs, const double interp_loc,
        TensorInterpolationErrorType& err_type);

    /*!
     * @name Get interpolation gradient, i.e. the derivative of the
     * interpolated matrix with respect to the interpolation location
     * vector / scalar.
     *
     * @note The derivative is computed numerically using a
     * perturbation approach with finite differences.
     *
     * @param[in]  ref_matrices  reference 3x3 matrices \f$ \boldsymbol{T}_j \f$ used as basis for
     *                            interpolation
     * @param[in]  ref_locs  locations \f$ \boldsymbol{x}_j \f$ of the reference matrices
     * @param[in]  interp_loc location \f$ \boldsymbol{x}_{\text{p}} \f$ of the interpolated
     * tensor
     * @param[in, out] err_type  error type of the tensor interpolator
     *  (shall be TensorInterpErrorType::NoErrors if no errors occurred)
     * @param[in]  perturbation_factor perturbation factor \f$
     * \epsilon_{\text{perturb}} \f$ to be used in the numerical
     * gradient determination (default: 1.0e-8).
     * @returns derivative of the interpolated matrix with respect to
     * the interpolation location vector. The result is a 9 x <dim>
     * matrix, where the second dimension corresponds to the dimension of
     * the location vector (1D, 2D, 3D).
     */
    //! @{
    /*!
     * @brief Standard method for interpolation locations with
     * variable dimensionality.
     */
    Core::LinAlg::Matrix<9, loc_dim> get_interpolation_gradient(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, TensorInterpolationErrorType& err_type,
        const double perturbation_factor = 1.0e-8);

    /*!
     * @brief Specialized method for 1D interpolation locations.
     * @note Calls the standard method but is more easy to handle when
     * using 1D locations.
     */
    Core::LinAlg::Matrix<9, 1> get_interpolation_gradient(
        const std::vector<Core::LinAlg::Matrix<3, 3>>& ref_matrices,
        const std::vector<double>& ref_locs, const double interp_loc,
        TensorInterpolationErrorType& err_type, const double perturbation_factor = 1.0e-8);
    //! @}

   private:
    /// polynomial space used for the interpolation of rotation vectors depending
    /// on the desired order (created in constructor call)
    Core::FE::PolynomialSpaceComplete<loc_dim, Core::FE::Polynomial> polynomial_space_;

    /// rotation interpolation type
    const RotationInterpolationType rot_interp_type_;

    /// eigenvalue interpolation type
    const EigenvalInterpolationType eigenval_interp_type_;

    /// interpolation parameters
    const ScalarInterpolationParams interp_params_;
  };

  /*!
   * @brief Perform polar decomposition \f$ \boldsymbol{T} = \boldsymbol{R} \boldsymbol{U} \f$ of
   * the 3x3 invertible matrix
   * \f$ \boldsymbol{T} $
   *
   * This method performs Step 1 of the procedure described in:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.5
   *
   *   Specifically, it splits a general tensor into its rotational and its stretch (symmetric,
   * positive definite) component. Moreover, the method calculates the eigenvalues, and it also
   * returns the spectral pairs of the tensor \f$ \boldsymbol{U} \f$, i.e., all 3 (eigenvalue,
   * eigenvector) eigenpairs. The spectral pairs are sorted in descending order of their
   * corresponding eigenvalues, while the eigenvalue matrix contains the lowest eigenvalue in (0,0)
   * and the highest in (2, 2).
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @param[out]  R_matrix  rotation matrix \boldsymbol{R}
   * @param[out]  U_matrix  stretch matrix \boldsymbol{U}
   * @param[out]  eigenval_matrix  eigenvalue matrix of the stretch matrix \boldsymbol{U}
   * @param[out]  spectral_pairs  vector of eigenpairs of the stretch matrix \boldsymbol{U}
   */
  void matrix_3x3_polar_decomposition(const Core::LinAlg::Matrix<3, 3>& inp_matrix,
      Core::LinAlg::Matrix<3, 3>& R_matrix, Core::LinAlg::Matrix<3, 3>& U_matrix,
      Core::LinAlg::Matrix<3, 3>& eigenval_matrix,
      std::array<std::pair<double, Core::LinAlg::Matrix<3, 1>>, 3>& spectral_pairs);

  /*!
   * @brief Compute the symmetric, positive-definite material stretch $\boldsymbol{U}$ from the
   * invertible matrix \f$ \boldsymbol{T} = \boldsymbol{R}
   * \boldsymbol{U} \f$
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @return  material stretch \boldsymbol{U}
   */
  Core::LinAlg::Matrix<3, 3> matrix_3x3_material_stretch(
      const Core::LinAlg::Matrix<3, 3>& inp_matrix);

  /*!
   * @brief Compute the symmetric, positive-definite spatial stretch $\boldsymbol{v}$ from the
   * invertible matrix \f$ \boldsymbol{T} = \boldsymbol{v}
   * \boldsymbol{R} \f$
   *
   * @param[in]  inp_matrix  input matrix \boldsymbol{T} to be decomposed
   * @return  spatial stretch \boldsymbol{v}
   */
  Core::LinAlg::Matrix<3, 3> matrix_3x3_spatial_stretch(
      const Core::LinAlg::Matrix<3, 3>& inp_matrix);


  /*!
   * @brief Calculate the rotation vector from a given rotation matrix, using Spurrier's algorithm
   *
   *
   * For further information, refer to:
   *    -# Spurrier, Comment on "Singularity-Free Extraction of a Quaternion from a
   * Direction-Cosine
   * Matrix", Journal of Spacecraft and Rockets 1978, 15(4):255-255
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.2
   * @param[in]  rot_matrix  input rotation matrix
   * @returns  corresponding rotation vector
   */
  Core::LinAlg::Matrix<3, 1> calc_rot_vect_from_rot_matrix(
      const Core::LinAlg::Matrix<3, 3>& rot_matrix);

  /*!
   * @brief Calculate the rotation vector (as 1-tensor) from a given rotation matrix (2-tensor),
   * using Spurrier's algorithm
   *
   *
   * For further information, refer to:
   *    -# Spurrier, Comment on "Singularity-Free Extraction of a Quaternion from a
   * Direction-Cosine
   * Matrix", Journal of Spacecraft and Rockets 1978, 15(4):255-255
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.2
   * @param[in]  rot_matrix  input rotation matrix
   * @returns  corresponding rotation vector
   */
  Core::LinAlg::Tensor<double, 3> calc_rotation_vector(
      Core::LinAlg::TensorView<const double, 3, 3> rot_matrix);

  /*!
   * @brief Calculate the rotation matrix from a given rotation vector, using the Rodrigues
   * formula
   *
   * For further information, refer to:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.1
   * @param[in]  rot_vect  input rotation vector
   * @returns  corresponding rotation matrix
   */
  Core::LinAlg::Matrix<3, 3> calc_rot_matrix_from_rot_vect(
      const Core::LinAlg::Matrix<3, 1>& rot_vect);

  /*!
   * @brief Calculate the rotation matrix (as 2-tensor) from a given rotation vector (1-tensor),
   * using Rodrigues' formula
   *
   * For further information, refer to:
   *    -# Satheesh et al., Structure-Preserving Invariant Interpolation Schemes for Invertible
   * Second-Order Tensors, Int J Numerical Methods Eng. 2024, 125, 10.1002/nme.7373, Section 2.2.1
   * @param[in]  rot_vect  input rotation vector
   * @returns  corresponding rotation 2-tensor
   */
  Core::LinAlg::Tensor<double, 3, 3> calc_rotation_matrix(
      Core::LinAlg::TensorView<const double, 3> rot_vect);

  namespace Internal
  {
    inline void make_eigenvectors_unique(
        std::span<const double> weights, std::span<Core::LinAlg::Tensor<double, 3, 3>> Q_i)
    {
      FOUR_C_ASSERT(!weights.empty(), "Input weights and eigenvectors must not be empty!");
      constexpr unsigned dim = 3;
      std::size_t closest_index = std::ranges::max_element(weights) - weights.begin();
      for (auto& t : Q_i)
      {
        // ensure that the first two eigenvectors have minimal rotation w.r.t. the closest
        // eigenvector matrix
        for (std::size_t i = 0; i < dim - 1; ++i)
        {
          double dot_product = 0.0;
          for (std::size_t col = 0; col < dim; ++col)
            dot_product += Q_i[closest_index](i, col) * t(i, col);

          if (dot_product < 0)
          {
            // flip the sign
            for (std::size_t col = 0; col < dim; ++col) t(i, col) = -t(i, col);
          }
        }

        // set the last eigenvector to ensure det(...) = 1 (cross-product of the first two)
        t(2, 0) = t(0, 1) * t(1, 2) - t(0, 2) * t(1, 1);
        t(2, 1) = t(0, 2) * t(1, 0) - t(0, 0) * t(1, 2);
        t(2, 2) = t(0, 0) * t(1, 1) - t(0, 1) * t(1, 0);
      }
    }
  }  // namespace Internal
}  // namespace Core::LinAlg

// Actual implementation of inline functions
Core::LinAlg::SymmetricTensor<double, 3, 3> Core::LinAlg::interpolate_spd(
    const std::ranges::sized_range auto& weights, const std::ranges::sized_range auto& tensors)
{
  FOUR_C_ASSERT(weights.size() == tensors.size(),
      "Number of weight entries must be equal to number of rotation tensors!");
  FOUR_C_ASSERT(!weights.empty(), "Input weights and rotation tensors must not be empty!");
  constexpr unsigned dim = 3;

  std::vector<Core::LinAlg::Tensor<double, dim, dim>> Q_i;
  std::vector<std::array<double, dim>> eigenvalues_i;
  Q_i.reserve(tensors.size());
  eigenvalues_i.reserve(tensors.size());

  for (const auto& t : tensors)
  {
    // perform eigenvalue decomposition
    const auto [eigenvalues, eigenvectors] = Core::LinAlg::eig(t);

    // Having degenerate eigenvalues may leave to weird interpolation results (don't allow for now)
    FOUR_C_ASSERT_ALWAYS(std::abs(eigenvalues[0] - eigenvalues[1]) > 1e-8 ||
                             std::abs(eigenvalues[1] - eigenvalues[2]) > 1e-8 ||
                             std::abs(eigenvalues[0] - eigenvalues[2]) > 1e-8,
        "Degenerate eigenvalues detected! The interpolation scheme requires distinct eigenvalues "
        "for proper operation.");

    // store eigenvectors and eigenvalues (The rows should be the eigenvectors, hence transpose it
    // here)
    Q_i.push_back(Core::LinAlg::transpose(eigenvectors));
    eigenvalues_i.push_back(eigenvalues);
  }

  // Note: At this stage, eigenvectors are not uniquely defined (sign ambiguity).
  Internal::make_eigenvectors_unique(weights, Q_i);

  // do interpolation of Q (same as for R)
  const Core::LinAlg::Tensor<double, dim, dim> Q_p = interpolate_rotation_tensor(weights, Q_i);

  // do eigenvalue interpolation
  const std::array<double, dim> interpolated_eigenvalues =
      log_interpolate_eigenvalues(weights, eigenvalues_i);

  // construct interpolated tensor from interpolated Q and eigenvalues
  return Core::LinAlg::assume_symmetry(
      Core::LinAlg::transpose(Q_p) *
      Core::LinAlg::TensorGenerators::diagonal(interpolated_eigenvalues) * Q_p);
}

Core::LinAlg::Tensor<double, 3, 3> Core::LinAlg::interpolate_rotation_tensor(
    const std::ranges::sized_range auto& weights,
    const std::ranges::sized_range auto& rotation_tensors)
{
  constexpr unsigned dim = 3;
  FOUR_C_ASSERT(weights.size() == rotation_tensors.size(),
      "Number of weight entries must be equal to number of rotation tensors!");
  FOUR_C_ASSERT(!weights.empty(), "Input weights and rotation tensors must not be empty!");
  FOUR_C_ASSERT(std::ranges::all_of(rotation_tensors, [](const auto& tensor)
                    { return std::abs(Core::LinAlg::det(tensor) - 1.0) < 1e-12; }),
      "All input tensors must be rotation tensors with det = +1!");

  // Transform to relative rotation tensors
  std::size_t closest_index = std::ranges::max_element(weights) - weights.begin();
  std::vector<Core::LinAlg::Tensor<double, dim, dim>> relative_rotation_tensors(
      rotation_tensors.size());

  std::transform(rotation_tensors.begin(), rotation_tensors.end(),
      relative_rotation_tensors.begin(), [&rotation_tensors, closest_index](const auto& R_i)
      { return Core::LinAlg::transpose(rotation_tensors[closest_index]) * R_i; });

  // Create rotation vector from relative rotation tensor
  std::vector<Core::LinAlg::Tensor<double, dim>> relative_rotation_vectors;
  relative_rotation_vectors.reserve(relative_rotation_tensors.size());

  std::transform(relative_rotation_tensors.begin(), relative_rotation_tensors.end(),
      std::back_inserter(relative_rotation_vectors),
      [](const auto& R_i) { return calc_rotation_vector(R_i); });

  // Interpolate rotation vectors
  Core::LinAlg::Tensor<double, dim> interpolated_rotation_vector =
      std::inner_product(weights.begin(), weights.end(), relative_rotation_vectors.begin(),
          Core::LinAlg::Tensor<double, dim>{}, std::plus<>(), std::multiplies<>());

  // compute rotation tensor
  return rotation_tensors[closest_index] * calc_rotation_matrix(interpolated_rotation_vector);
}

std::array<double, 3> Core::LinAlg::log_interpolate_eigenvalues(
    const std::ranges::sized_range auto& weights, const std::ranges::sized_range auto& eigenvalues)
{
  constexpr unsigned dim = 3;
  FOUR_C_ASSERT(weights.size() == eigenvalues.size(),
      "Number of weight entries must be equal to number of eigenvalue sets!");
  std::array<double, dim> interpolated_eigenvalues;

  for (std::size_t i = 0; i < dim; ++i)
  {
    double log_eigenvalue = 0.0;
    for (std::size_t j = 0; j < weights.size(); ++j)
    {
      FOUR_C_ASSERT(eigenvalues[j][i] > 0.0,
          "All eigenvalues must be positive for logarithmic eigenvalue interpolation!");
      log_eigenvalue += weights[j] * std::log(eigenvalues[j][i]);
    }
    interpolated_eigenvalues[i] = std::exp(log_eigenvalue);
  }

  return interpolated_eigenvalues;
}

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTLIS_QUATERNION_INTERPOLATION_HPP
#define FOUR_C_LINALG_UTLIS_QUATERNION_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_utils_scalar_interpolation.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core
{
  namespace LinAlg
  {
    /**
     * @brief Spherical linear interpolation (SLERP) between two quaternions.
     *
     * @param q1 The first normalized quaternion.
     * @param q2 The second normalized quaternion.
     * @param t Interpolation parameter in [0, 1].
     * @return Interpolated quaternion (normalized).
     */
    Core::LinAlg::Matrix<4, 1> spherical_linear_interpolation(
        const Core::LinAlg::Matrix<4, 1>& q1, const Core::LinAlg::Matrix<4, 1>& q2, double t);

    /**
     * @brief Generalized Spherical Linear Interpolator for quaternions.
     *
     * This class implements a generalized spherical linear interpolation (SLERP) method for
     * quaternions, allowing for interpolation based on reference locations and weights.
     * It supports different dimensions for reference locations, enabling flexible use cases.
     *
     * The interpolation is based on the method (A2) described by Buss and Fillmore, 2001.
     */
    template <unsigned int loc_dim = 1>
    class GeneralizedSphericalLinearInterpolator
    {
     public:
      /**
       * @brief Constructs a GeneralizedSphericalLinearInterpolator for quaternion interpolation.
       *
       * This constructor initializes the interpolator with a set of unit quaternions,
       * their corresponding reference locations, a weighting function, and interpolation
       * parameters. The interpolator can be used to perform smooth interpolation between
       * orientations represented by quaternions, generalized to arbitrary reference locations.
       *
       * @param unit_quaternions A vector of unit quaternions (as 4x1 matrices) to
       * interpolate between.
       * @param ref_locs A vector of reference locations (as loc_dim x 1 matrices) associated with
       * each quaternion.
       * @param weight_func The weighting function used to compute interpolation weights based on
       * reference locations.
       * @param interp_params Parameters controlling the weighting function.
       */
      GeneralizedSphericalLinearInterpolator(
          const std::vector<Core::LinAlg::Matrix<4, 1>>& unit_quaternions,
          const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
          const Core::LinAlg::ScalarInterpolation::WeightingFunction weight_func,
          const Core::LinAlg::ScalarInterpolation::InterpParams& interp_params);

      /**
       * @brief Constructs a GeneralizedSphericalLinearInterpolator with a set of normalized
       * quaternions and corresponding weights.
       *
       * This constructor initializes the interpolator using a vector of unit quaternions
       * and their associated weights. The quaternions should be 4-dimensional column vectors, and
       * all quaternions must be normalized to ensure correct interpolation. The weights determine
       * the influence of each quaternion in the interpolation process.
       *
       * @param unit_quaternions A vector of unit quaternions (4x1 matrices) to be
       * used in the interpolation.
       * @param normalized_weights A vector of weights corresponding to each quaternion. The size
       * of this vector must match the number of quaternions.
       *
       * @note The sum of the weights must be 1 and all weights should be non-negative.
       * @throws std::invalid_argument if the sizes of the input vectors do not match or if any
       * quaternion is not normalized.
       */
      GeneralizedSphericalLinearInterpolator(
          const std::vector<Core::LinAlg::Matrix<4, 1>>& unit_quaternions,
          const std::vector<double>& normalized_weights);

      /**
       * @brief Computes and returns an interpolated quaternion.
       *
       * This function calculates an interpolated quaternion, typically used for smooth
       * transitions or rotations in 3D space. Optionally, it can take a pointer to a location
       * vector to specify the interpolation location.
       *
       * @param interp_loc Optional pointer to a location vector of dimension `loc_dim`.
       *                   Must be provided for constructor with weight calculation within the
       * class and nullptr for constructor with weights.
       * @return Core::LinAlg::Matrix<4, 1> The resulting interpolated quaternion as a 4x1 matrix.
       */
      Core::LinAlg::Matrix<4, 1> get_interpolated_quaternion(
          Core::LinAlg::Matrix<loc_dim, 1>* interp_loc = nullptr);

     private:
      /**
       * @brief Computes and returns an initial estimate for a quaternion.
       *
       * This function generates a 4x1 matrix representing the initial guess
       * for a quaternion, which can be used as a starting point in quaternion
       * interpolation.
       *
       * @return Core::LinAlg::Matrix<4, 1> The initial quaternion estimate as a 4x1 matrix.
       */
      Core::LinAlg::Matrix<4, 1> initial_estimate();

      /**
       * @brief Projects the difference between two quaternions onto the subspace orthogonal to
       * the second quaternion.
       *
       * Given two quaternions, this function computes the vector (quaternion_1 - quaternion_2)
       * and then removes its component along quaternion_2, effectively projecting the difference
       * onto the hyperplane orthogonal to quaternion_2.
       *
       * @param quaternion_1 The first quaternion (as a 4x1 matrix).
       * @param quaternion_2 The second quaternion (as a 4x1 matrix), which defines the direction
       * to project orthogonally to.
       * @return Core::LinAlg::Matrix<4, 1> The projected quaternion difference, orthogonal to
       * quaternion_2.
       */
      Core::LinAlg::Matrix<4, 1> project_quaternion(const Core::LinAlg::Matrix<4, 1>& quaternion_1,
          const Core::LinAlg::Matrix<4, 1>& quaternion_2);

      /**
       * @brief Rotates a quaternion by a given direction using generalized spherical linear
       * interpolation.
       *
       * This function modifies the input quaternion by rotating it according to the provided
       * direction vector. The direction vector is assumed to be orthogonal to the quaternion and
       * its norm represents the rotation angle.
       *
       * @param quaternion [in,out] The quaternion to be rotated. Must be a unit quaternion (norm
       * \approx 1).
       * @param direction [in] The direction vector representing the rotation axis and angle.
       *
       * @note
       * - The function asserts that the input quaternion is normalized and that the direction is
       * orthogonal to the quaternion.
       * - If the direction vector is zero, the quaternion remains unchanged.
       * - The rotation is performed using the axis-angle representation encoded in the direction
       * vector.
       */
      void rotate_quaternion(
          Core::LinAlg::Matrix<4, 1>& quaternion, const Core::LinAlg::Matrix<4, 1>& direction);

      /**
       * @brief Computes a 4x4 orthonormal basis matrix using the Gram-Schmidt process from a
       * given quaternion.
       *
       * This function takes a quaternion represented as a 4x1 matrix and generates a 4x4 matrix
       * whose columns form an orthonormal basis in 4D space. The first column of the resulting
       * matrix corresponds to the normalized input quaternion, and the remaining columns are
       * constructed to be orthogonal to each other and to the first column, ensuring the entire
       * matrix is orthonormal.
       *
       * @param quaternion A 4x1 matrix representing the input quaternion to be used as the first
       * basis vector.
       * @return Core::LinAlg::Matrix<4, 4> A 4x4 matrix whose columns form an orthonormal basis
       * in 4D space.
       */
      Core::LinAlg::Matrix<4, 4> gram_schmidt_orthonormal_basis(
          const Core::LinAlg::Matrix<4, 1>& quaternion);

      /**
       * @brief Computes a right-handed orthonormal basis given a unit vector u.
       *
       * Given an input vector u, this function computes two additional vectors v and w such that
       * {u, v, w} forms a right-handed orthonormal basis in 3D space. The vector v is chosen to
       * be orthogonal to u and is constructed based on the components of u to avoid degeneracy.
       * The vector w is computed as the cross product of u and v, and both v and w are normalized
       * to ensure orthonormality.
       *
       * @param[in]  u Input vector (should be normalized).
       * @param[out] v Output vector, orthogonal to u and normalized.
       * @param[out] w Output vector, orthogonal to both u and v, normalized (u x v).
       */
      void right_orthonormal_basis(Core::LinAlg::Matrix<3, 1>& u, Core::LinAlg::Matrix<3, 1>& v,
          Core::LinAlg::Matrix<3, 1>& w);

      /**
       * @brief Computes the weighted sum of a set of quaternions.
       *
       * This function takes a vector of quaternions and a corresponding vector of weights,
       * and returns their weighted sum as a quaternion.
       *
       * @param quaternions A vector containing the quaternions to be weighted and summed.
       * @param weights A vector of weights corresponding to each quaternion. The size of this
       *        vector must match the size of the quaternions vector.
       * @return The resulting normalized weighted sum quaternion as a 4x1 matrix.
       *
       */
      Core::LinAlg::Matrix<4, 1> weighted_sum(
          const std::vector<Core::LinAlg::Matrix<4, 1>>& quaternions,
          const std::vector<double>& weights);

      // Data members
      // The following members store the unit quaternions, their weights, reference
      // locations, the weighting function, and interpolation parameters used in the
      // interpolation process.
      std::vector<Core::LinAlg::Matrix<4, 1>> unit_quaternions_;
      std::vector<double> normalized_weights_;
      std::vector<Core::LinAlg::Matrix<loc_dim, 1>> ref_locs_;
      Core::LinAlg::ScalarInterpolation::WeightingFunction weight_func_;
      Core::LinAlg::ScalarInterpolation::InterpParams interp_params_;
    };
  }  // namespace LinAlg
}  // namespace Core

FOUR_C_NAMESPACE_CLOSE

#endif
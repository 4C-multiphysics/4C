// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_LINALG_UTILS_SCALAR_INTERPOLATION_HPP
#define FOUR_C_LINALG_UTILS_SCALAR_INTERPOLATION_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"

#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Core::LinAlg
{
  /// enum class for the interpolation type
  enum class ScalarInterpolationType
  {
    logarithmic_weighted_average,      ///< logarithmic weighted average,
    moving_least_squares,              ///< moving least squares,
    logarithmic_moving_least_squares,  ///< logarithmic moving least squares,
  };

  /// enum class for the interpolation weighting function
  enum class ScalarInterpolationWeightingFunction
  {
    inverse_distance,  ///< inverse distance weighting
    exponential,       ///< exponential weighting
    unity,             ///< equal weighting
  };

  /// struct containing parameters used for the scalar interpolation
  struct ScalarInterpolationParams
  {
    /// exponential decay factor of the weighting functions, as shown in Satheesh,
    /// 2024, 10.1002/nme.7373, Eq. (21)
    double exponential_decay_c = 10.0;

    /// inverse distance power
    /// used for the inverse distance weighting function
    /// (i.e., 1 / (distance^power))
    std::optional<double> inverse_distance_power;

    /// distance threshold below which a reference point is considered coincident with the
    /// interpolation point; if any reference point is within this threshold, its weight is set
    /// to 1.0 and all others to 0.0
    double distance_threshold = 1.0e-8;
  };

  /**
   * @brief Computes the polynomial shape function values at a given location.
   *
   * This templated function evaluates the shape function for a polynomial of specified order
   * at a given location in the local coordinate system.
   *
   * @tparam loc_dim Dimension of the local coordinate system.
   * @tparam poly_order Order of the polynomial shape function.
   * @tparam num_coefficients Number of coefficients (basis functions) in the polynomial, this is
   * important in the context of an incomplete polynomial basis. Example: An incomplete quadratic
   * basis: [1,x,y,x^2], missing xy and y^2
   * @param location The local coordinates where the shape function is to be evaluated.
   * @return A column vector containing the values of the shape function basis at the given
   * location.
   */
  template <unsigned int loc_dim, unsigned int poly_order, unsigned int num_coefficients>
  Core::LinAlg::Matrix<num_coefficients, 1> polynomial_shape_function(
      const Core::LinAlg::Matrix<loc_dim, 1>& location);

  /**
   * @brief Calculates normalized interpolation weights for a given interpolation location.
   *
   * This function computes the weights for each reference location based on the provided
   * weighting function and interpolation parameters, and normalizes them so that their sum
   * is 1.
   *
   * @param ref_locs A vector of reference locations
   * @param interp_loc The interpolation location
   * @param weight_func The weighting function used to compute the weights based on distance or
   * other criteria.
   * @param interp_params Additional parameters required by the weighting function.
   * @return std::vector<double> A vector of normalized weights corresponding to each reference
   * location.
   */
  template <unsigned int loc_dim>
  std::vector<double> calculate_normalized_weights(
      const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
      const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc,
      const ScalarInterpolationWeightingFunction weight_func,
      const ScalarInterpolationParams& interp_params);


  /**
   * @class ScalarInterpolator
   * @brief A class template for performing scalar interpolation using various methods and
   * weighting functions.
   *
   * This class provides functionality to interpolate scalar values at arbitrary locations based
   * on reference data points and their associated locations. It supports different interpolation
   * types, including moving least squares (MLS), logarithmic interpolation, and log-domain MLS,
   * with customizable weighting functions and interpolation parameters.
   *
   * @tparam loc_dim The spatial dimension of the interpolation (e.g., 2 for 2D, 3 for 3D).
   * @tparam poly_order The polynomial order used in the interpolation basis (MLS &LOGMLS).
   * @tparam num_coefficients The number of coefficients in the interpolation basis, applying to
   * the polynomial shape function (MLS & LOGMLS).
   *
   * Usage:
   *   - Construct with the desired interpolation type, weighting function, and parameters.
   *      for example:
   *        (a) LOG: (second and third template argument are ignored)
   *          ScalarInterpolator<3> interpolator(ScalarInterpolationType::MLS,
   *          ScalarInterpolationWeightingFunction::exponential, interp_params)
   *
   *        (b) MLS/LOGMLS: (all three template arguments are used)
   *          ScalarInterpolator<3, 2, 10>
   *            interpolator( ScalarInterpolationType::MLS,
   * ScalarInterpolationWeightingFunction::exponential, interp_params);
   *   - Use get_interpolated_scalar() to compute interpolated values at a given location.
   *   - Use calculate_normalized_weights() to obtain normalized weights for reference points, if
   * interested to return weights.
   *
   * Private helper methods implement the core interpolation algorithms, including MLS,
   * logarithmic variants, and log MLS.
   */
  template <unsigned int loc_dim, unsigned int poly_order = 1, unsigned int num_coefficients = 1>
  class ScalarInterpolator
  {
   public:
    /**
     * @brief Constructs a ScalarInterpolator with the specified interpolation type, weighting
     * function, and parameters.
     *
     * @param scalar_interp_type The type of scalar interpolation to use.
     * @param weight_func The weighting function to apply during interpolation.
     * @param interp_params Additional parameters required for the interpolation process.
     */
    ScalarInterpolator(const ScalarInterpolationType scalar_interp_type,
        const ScalarInterpolationWeightingFunction weight_func,
        const ScalarInterpolationParams& interp_params);

    /**
     * @brief Interpolates scalar values at a given location using provided scalar data and
     * reference locations.
     *
     * This function computes the interpolated scalar values at the specified interpolation
     * location @p interp_loc based on the input scalar data and their corresponding reference
     * locations.
     *
     * @param scalar_data A vector of vectors containing scalar values at each reference location.
     * @param ref_locs A vector of reference locations.
     * @param interp_loc The location at which the scalar value is to be interpolated.
     * @return std::vector<double> The interpolated scalar values at the specified interpolation
     * location.
     */
    std::vector<double> get_interpolated_scalar(const std::vector<std::vector<double>>& scalar_data,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc);

   private:
    /**
     * @brief Computes the moving least squares (MLS) interpolation for a given location.
     *
     * This function performs MLS interpolation using the provided scalar data, reference
     * locations, interpolation location, and associated weights. It returns the interpolated
     * scalar values.
     *
     * @param scalar_data A vector of vectors containing scalar data at each reference location.
     * @param ref_locs A vector of reference locations.
     * @param interp_loc The location at which to interpolate.
     * @param weights A vector of weights corresponding to each reference location, typically
     * based on distance to interp_loc.
     * @return std::vector<double> The interpolated scalar values at the specified interpolation
     * location.
     */
    std::vector<double> moving_least_square(const std::vector<std::vector<double>>& scalar_data,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, const std::vector<double>& weights);

    /**
     * @brief Computes the logarithmic interpolation of scalar data using the provided weights.
     *
     * This function takes a 2D vector of scalar data and a vector of weights, and returns a
     * vector containing the logarithmic interpolation results for each set of scalar values.
     *
     * @param scalar_data A 2D vector where each inner vector contains scalar values to be
     * interpolated.
     * @param weights A vector of weights corresponding to the scalar values for interpolation.
     * @return A vector of doubles representing the logarithmic interpolation results.
     */
    std::vector<double> logarithmic_weighted_average(
        const std::vector<std::vector<double>>& scalar_data, const std::vector<double>& weights);

    /**
     * @brief Computes the log moving least squares interpolation for scalar data.
     *
     * This function performs moving least squares interpolation in the logarithmic domain
     * for a set of scalar data points. It uses the provided reference locations, interpolation
     * location, and associated weights to compute the interpolated values.
     *
     * @param scalar_data A vector of vectors containing the scalar data values at each reference
     * location.
     * @param ref_locs A vector of reference locations.
     * @param interp_loc The location at which the interpolation is to be performed.
     * @param weights A vector of weights corresponding to each reference location.
     * @return std::vector<double> The interpolated scalar values at the specified interpolation
     * location.
     */
    std::vector<double> logarithmic_moving_least_squares(
        const std::vector<std::vector<double>>& scalar_data,
        const std::vector<Core::LinAlg::Matrix<loc_dim, 1>>& ref_locs,
        const Core::LinAlg::Matrix<loc_dim, 1>& interp_loc, const std::vector<double>& weights);

    const ScalarInterpolationType scalar_interp_type_;
    const ScalarInterpolationWeightingFunction weight_func_;
    const ScalarInterpolationParams interp_params_;
  };
}  // namespace Core::LinAlg

FOUR_C_NAMESPACE_CLOSE

#endif

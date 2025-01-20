// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_MUSCLE_UTILS_HPP
#define FOUR_C_MAT_MUSCLE_UTILS_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_utils_function.hpp"

#include <unordered_map>
#include <vector>

FOUR_C_NAMESPACE_OPEN

namespace Mat::Utils::Muscle
{
  /*!
   * @brief Evaluate Lambert W function with Halley's method
   *
   * Solution of Lambert W function is functional inverse of xi = W_0*exp(W_0)
   * Computation here restricted to principal branch W_0
   * Use of Halley's method according to:
   * https://blogs.mathworks.com/cleve/2013/09/02/the-lambert-w-funsolutiction/
   *
   * @param[in]     xi Argument of Lambert W function W(xi)
   * @param[in,out] WO Solution of principal branch of Lambert W function
   * @param[in]     tol Tolerance for Halley's approximation
   * @param[in]     maxiter Maximal number of iterations
   */
  void evaluate_lambert(const double xi, double& W0, const double tol, const int maxiter);

  /*!
   * @brief Evaluate the force-stretch dependency according to Ehret et al.
   *
   * Reference: A. E. Ehret, M. Boel, M. Itskova, 'A continuum constitutive model for the active
   * behaviour of skeletal muscle', Journal of the Mechanics and Physics of Solids, vol. 59, pp.
   * 625-636, 2011, doi: 10.1016/j.jmps.2010.12.008.
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaMin Minimal fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[out]    fxi Force-stretch function result between zero and one
   */
  double evaluate_force_stretch_dependency_ehret(
      const double lambdaM, const double lambdaMin, const double lambdaOpt);

  /*!
   * @brief Evaluate the derivative of the force-stretch dependency according to Ehret et al.
   *        w.r.t. the fiber stretch
   *
   * Reference: A. E. Ehret, M. Boel, M. Itskova, 'A continuum constitutive model for the active
   * behaviour of skeletal muscle', Journal of the Mechanics and Physics of Solids, vol. 59, pp.
   * 625-636, 2011, doi: 10.1016/j.jmps.2010.12.008.
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaMin Minimal fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[out]    dFxidLambdaM Derivative of the force-stretch function w.r.t. the fiber
   *                            stretch
   */
  double evaluate_derivative_force_stretch_dependency_ehret(
      const double lambdaM, const double lambdaMin, const double lambdaOpt);

  /*!
   * @brief Evaluate the integral of the force-stretch dependency according to Giantesio et al.
   *        w.r.t. the fiber stretch
   *
   * Reference: G. Giantesio, A. Musesti, 'Strain-dependent internal parameters in hyperelastic
   * biological materials', International Journal of Non-Linear Mechanics, vol. 95, pp. 162-167,
   * 2017, doi:10.1016/j.ijnonlinmec.2017.06.012.
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaMin Minimal fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[out]    intFxi Integral of the force-stretch function w.r.t. the fiber stretch
   */
  double evaluate_integral_force_stretch_dependency_ehret(
      const double lambdaM, const double lambdaMin, const double lambdaOpt);

  /*!
   * @brief Evaluate the force-velocity dependency in the style of Boel et al.
   *
   * Force-velocity dependency according to Boel et al. with some slight modifications. Boel et.
   * al. scale and shift the eccentric branch of the velocity dependency using the parameter d.
   * To be able to turn the velocity dependency on and off, this concept is adopted for the
   * concentric branch as well. Choosing de=dc=0 leads to a neglection of the velocity
   * dependence. Choosing dc=1 and de=d-1 reproduces the function presented by Boel et al..
   *
   * Reference: M. Boel and S. Reese, 'Micromechanical modelling of skeletal muscles based on
   * the finite element method', Computer Methods in Biomechanics and Biomedical Engineering,
   * vol. 11, no. 5, pp. 489-504, 2008, doi: 10.1080/10255840701771750.
   *
   * @param[in]     dotLambdaM Contraction velocity / stretch rate
   * @param[in]     dotLambdaMMin Minimal stretch rate
   * @param[in]     de Amplitude of the eccentric velocity dependency function
   * @param[in]     dc Amplitude of the concentric velocity dependency function
   * @param[in]     ke Curvature of the eccentric velocity dependency function
   * @param[in]     kc Curvature of the concentric velocity dependency function
   * @param[out]    fv Force-velocity function result between zero and one
   */
  double evaluate_force_velocity_dependency_boel(const double dotLambdaM,
      const double dotLambdaMMin, const double de, const double dc, const double ke,
      const double kc);

  /*!
   * @brief Evaluate the derivative of the force-velocity dependency in the style of Boel et al.
   *        w.r.t. the fiber stretch
   *
   * Force-velocity dependency according to Boel et al. with some slight modifications. Boel et.
   * al. scale and shift the eccentric branch of the velocity dependency using the parameter d.
   * To be able to turn the velocity dependency on and off, this concept is adopted for the
   * concentric branch as well. Choosing de=dc=0 leads to a neglection of the velocity
   * dependence. Choosing dc=1 and de=d-1 reproduces the function presented by Boel et al..
   *
   * Reference: M. Boel and S. Reese, 'Micromechanical modelling of skeletal muscles based on
   * the finite element method', Computer Methods in Biomechanics and Biomedical Engineering,
   * vol. 11, no. 5, pp. 489-504, 2008, doi: 10.1080/10255840701771750.
   *
   * @param[in]     dotLambdaM Contraction velocity / stretch rate
   * @param[in]     dDotLambdaMdLambdaM Derivative of contraction velocity w.r.t. the fiber
   *                                    stretch
   * @param[in]     dotLambdaMMin Minimal stretch rate
   * @param[in]     de Amplitude of the eccentric velocity dependency function
   * @param[in]     dc Amplitude of the concentric velocity dependency function
   * @param[in]     ke Curvature of the eccentric velocity dependency function
   * @param[in]     kc Curvature of the concentric velocity dependency function
   * @param[out]    dFvdLambdaM Derivative of the force-velocity function w.r.t. the fiber
   *                           stretch
   */
  double evaluate_derivative_force_velocity_dependency_boel(const double dotLambdaM,
      const double dDotLambdaMdLambdaM, const double dotLambdaMMin, const double de,
      const double dc, const double ke, const double kc);

  /*!
   * @brief Evaluate the time-dependent optimal active stress by summation single twitches
   * according to Ehret et al.
   *
   * Reference: A. E. Ehret, M. Boel, M. Itskova, 'A continuum constitutive model for the active
   * behaviour of skeletal muscle', Journal of the Mechanics and Physics of Solids, vol. 59, pp.
   * 625-636, 2011, doi: 10.1016/j.jmps.2010.12.008.
   *
   * @param[in]     Na Number of active motor units (MU) per undeformed muscle cross-sectional
   *                   area
   * @param[in]     muTypesNum Number of motor unit (MU) types
   * @param[in]     rho Fraction of MU types
   * @param[in]     I Interstimulus intervals of MU types
   * @param[in]     F Twitch forces of MU types
   * @param[in]     T Twitch contraction times of MU types
   * @param[in]     actIntervalsNum Number of time intervals where activation is prescribed
   * @param[in]     actTimes Time boundaries between intervals
   * @param[in]     actValues Scaling factor in intervals (1=full activation, 0=no activation)
   * @param[in]     currentTime Current time
   * @param[out]    Poptft Time-dependent optimal active stress at currentTime
   */
  double evaluate_time_dependent_active_stress_ehret(const double Na, const int muTypesNum,
      const std::vector<double>& rho, const std::vector<double>& I, const std::vector<double>& F,
      const std::vector<double>& T, const int actIntervalsNum, const std::vector<double>& actTimes,
      const std::vector<double>& actValues, const double currentTime);

  /*!
   * @brief Evaluate the active force-stretch dependency according to Blemker et al.
   *
   * Reference: S. S. Blemker, P. M. Pinsky und S. L. Delp, 'A 3D model of muscle reveals the
   * causes of nonuniform strains in the biceps brachii', Journal of biomechanics, vol. 38, no.
   * 4, pp. 657-665, 2005. doi: 10.1016/j.jbiomech.2004.04.009
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[out]    fxi Force-stretch function result between zero and one
   */
  double evaluate_active_force_stretch_dependency_blemker(
      const double lambdaM, const double lambdaOpt);

  /*!
   * @brief Evaluate the derivative of the active force-stretch dependency according to
   *        Blemker et al. w.r.t. the fiber stretch
   *
   * Reference: S. S. Blemker, P. M. Pinsky und S. L. Delp, 'A 3D model of muscle reveals the
   * causes of nonuniform strains in the biceps brachii', Journal of biomechanics, vol. 38, no.
   * 4, pp. 657-665, 2005. doi: 10.1016/j.jbiomech.2004.04.009
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[out]    dFxidLambdaM Derivative of the force-stretch function w.r.t. the fiber
   *                            stretch
   */
  double evaluate_derivative_active_force_stretch_dependency_blemker(
      const double lambdaM, const double lambdaOpt);

  /*!
   * @brief Evaluate the passive force-stretch dependency according to Blemker et al.
   *
   * Reference: S. S. Blemker, P. M. Pinsky und S. L. Delp, 'A 3D model of muscle reveals the
   * causes of nonuniform strains in the biceps brachii', Journal of biomechanics, vol. 38, no.
   * 4, pp. 657-665, 2005. doi: 10.1016/j.jbiomech.2004.04.009
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[in]     lambdaStar Fiber stretch where normalized passive fiber force becomes linear
   * @param[in]     P1 Linear material parameter for along-fiber response
   * @param[in]     P2 Exponential material parameter for along-fiber response
   * @param[out]    fxi Force-stretch function result between zero and one
   */
  double evaluate_passive_force_stretch_dependency_blemker(const double lambdaM,
      const double lambdaOpt, const double lambdaStar, const double P1, const double P2);

  /*!
   * @brief Evaluate the derivative of the passive force-stretch dependency according to
   *        Blemker et al. w.r.t. the fiber stretch
   *
   * Reference: S. S. Blemker, P. M. Pinsky und S. L. Delp, 'A 3D model of muscle reveals the
   * causes of nonuniform strains in the biceps brachii', Journal of biomechanics, vol. 38, no.
   * 4, pp. 657-665, 2005. doi: 10.1016/j.jbiomech.2004.04.009
   *
   * @param[in]     lambdaM Fiber stretch
   * @param[in]     lambdaOpt Optimal fiber stretch
   * @param[in]     lambdaStar Fiber stretch where normalized passive fiber force becomes linear
   * @param[in]     P1 Linear material parameter for along-fiber response
   * @param[in]     P2 Exponential material parameter for along-fiber response
   * @param[out]    dFxidLambdaM Derivative of the force-stretch function w.r.t. the fiber
   *                            stretch
   */
  double evaluate_derivative_passive_force_stretch_dependency_blemker(const double lambdaM,
      const double lambdaOpt, const double lambdaStar, const double P1, const double P2);

  /*!
   * @brief Evaluate the time-dependent optimal (i.e. maximal) active stress using a
   * tanh-function
   *
   * The time-dependent activation function is computed for any t_tot>t_act_start as
   * ft = alpha*tanh(beta*(t_tot-t_act_start)).
   *
   * The time-dependent optimal active stress is obtained by sigma_opt = sigma_max * ft
   *
   * @param[in]     sigma_max Optimal (i.e. maximal) active stress
   * @param[in]     alpha Tetanised activation level
   * @param[in]     beta Scaling factor
   * @param[in]     t_act_start Time of start of activation
   * @param[in]     t_current Current time
   * @param[out]    sigma_max_ft Time-dependent optimal active stress at t_current
   */
  double evaluate_time_dependent_active_stress_tanh(const double sigma_max, const double alpha,
      const double beta, const double t_act_start, const double t_current);

  /*!
   * @brief Evaluate the time- and space-dependent optimal (i.e. maximal) active stress through
   * any arbitrary analytical function ft defined in the input file (e.g.,
   * SYMBOLIC_FUNCTION_OF_SPACE_TIME x*tanh(10*t)).
   *
   * The time- and space-dependent activation function ft is computed by evaluating the given
   * activation_function in x and t_current.
   *
   * The time-dependent optimal active stress is obtained by sigma_opt = sigma_max * ft
   *
   * @param[in]     sigma_max Optimal (i.e. maximal) active stress
   * @param[in]     activation_function Time-/space-dependent function to be evaluated
   * @param[in]     t_current Current time
   * @param[in]     x Point in 3D space (e.g., element centroid)
   * @param[out]    sigma_max_ft Time-/space-dependent optimal active stress at x, t_current
   */
  double evaluate_time_space_dependent_active_stress_by_funct(const double sigma_max,
      const Core::Utils::FunctionOfSpaceTime& activation_function, const double t_current,
      const Core::LinAlg::Matrix<3, 1>& x);

  /*!
   * @brief Evaluate the time- and space-dependent optimal (i.e. maximal) active stress through
   * a map with element-wise defined activation values. The path to the file defining the map is
   * given as the parameter MAP_FILE. The file specifies activation values at corresponding times
   * for each element id. Those values need to be definited according to the following line string
   * pattern: global element id: time_0, activation_0; time_1, activation_1; ....
   *
   * The time- and space-dependent activation function ft is computed by accessing the
   * activation_map for a given element id and time. Evaluating the activation_map at the element id
   * (key) returns a vector of pairs (value). Those vector entries correspond to a time-"activation
   * value"-pair. The activation at the current time t_current is found via linear interpolation
   * between the prescribed time-"activation value"-pairs.
   *
   * The time-dependent optimal active stress is obtained by sigma_opt = sigma_max * ft
   *
   * @param[in]     sigma_max Optimal (i.e. maximal) active stress
   * @param[in]     activation_map Map to be evaluated
   * @param[in]     t_current Current time
   * @param[in]     activation_map_key Global element id serving as key for the defined mapping
   * @param[out]    sigma_max_ft Time-/space-dependent optimal active stress for the given element
   *                             id and t_current
   */
  double evaluate_time_space_dependent_active_stress_by_map(const double sigma_max,
      const std::unordered_map<int, std::vector<std::pair<double, double>>>& activation_map,
      const double t_current, const int activation_map_key);

  /*!
   *  @brief Returns the fiber stretch in the current configuration
   *
   *  @param[in] C Cauchy-Green strain tensor
   *  @param[in] M Structural tensor of fiber directions
   *  @param[out] lambdaM Fiber stretch
   */
  double fiber_stretch(const Core::LinAlg::Matrix<3, 3>& C, const Core::LinAlg::Matrix<3, 3>& M);

  /*!
   *  @brief Returns the derivative of the fiber stretch w.r.t. the Cauchy-Green strain
   *
   *  @param[in] lambdaM Fiber stretch
   *  @param[in] C Cauchy-Green strain tensor
   *  @param[in] M Structural tensor of fiber directions
   *  @param[out] dlambdaMdC Derivative of the fiber stretch w.r.t. the Cauchy-Green strains
   */
  Core::LinAlg::Matrix<3, 3> d_fiber_stretch_dc(const double lambdaM,
      const Core::LinAlg::Matrix<3, 3>& C, const Core::LinAlg::Matrix<3, 3>& M);

  /*!
   *  @brief Returns the contraction velocity computed by a Backward Euler approximation
   *
   *  @param[in] lambdaM Fiber stretch at time t_n
   *  @param[in] lambdaMOld Fiber stretch at time t_{n-1}
   *  @param[in] timeStepSize Timestep size t_n - t_{n-1}
   *  @return dotLambdaM Contraction velocity, i.e. time derivative of the fiber stretch
   */
  double contraction_velocity_bw_euler(
      const double lambdaM, const double lambdaMOld, const double timeStepSize);
}  // namespace Mat::Utils::Muscle

FOUR_C_NAMESPACE_CLOSE

#endif

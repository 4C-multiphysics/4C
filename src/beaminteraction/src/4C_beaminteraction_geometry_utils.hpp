// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_BEAMINTERACTION_GEOMETRY_UTILS_HPP
#define FOUR_C_BEAMINTERACTION_GEOMETRY_UTILS_HPP

#include "4C_config.hpp"

#include "4C_fem_general_element.hpp"
#include "4C_linalg_fixedsizematrix.hpp"

FOUR_C_NAMESPACE_OPEN


namespace BeamInteraction
{
  namespace Geo
  {
    // point-to-curve projection: solve minimal distance problem
    // convergence criteria for local Newton's method
    const unsigned int POINT_TO_CURVE_PROJECTION_MAX_NUM_ITER = 50;
    const double POINT_TO_CURVE_PROJECTION_TOLERANCE_RESIDUUM = 1.0e-10;
    const double POINT_TO_CURVE_PROJECTION_TOLERANCE_INCREMENT = 1.0e-10;
    // threshold values for sanity checks
    const double POINT_TO_CURVE_PROJECTION_IDENTICAL_POINTS_TOLERANCE = 1.0e-12;
    const double POINT_TO_CURVE_PROJECTION_NONUNIQUE_MINIMAL_DISTANCE_TOLERANCE = 1.0e-12;

    /** \brief solves minimal distance problem to find the closest point on a 3D spatial curve
     *         (i.e. its curve parameter value) relative to a given point
     *         a.k.a 'unilateral' closest-point projection
     *
     */
    template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
    bool point_to_curve_projection(Core::LinAlg::Matrix<3, 1, T> const& r_slave, T& xi_master,
        double const& xi_master_initial_guess,
        const Core::LinAlg::Matrix<3 * numnodes * numnodalvalues, 1, T>&
            master_centerline_dof_values,
        const Core::FE::CellType& master_distype, double master_ele_ref_length);

    /** \brief evaluates residual of orthogonality condition for so-called unilateral closest-point
     *         projection, i.e. a point-to-curve projection
     *
     */
    template <typename T>
    void evaluate_point_to_curve_orthogonality_condition(T& f,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master);

    /** \brief evaluates Jacobian of orthogonality condition for so-called unilateral closest-point
     *         projection, i.e. a point-to-curve projection
     *
     */
    template <typename T>
    bool evaluate_linearization_point_to_curve_orthogonality_condition(T& df,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r, const double norm_delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixi_master);

    /** \brief compute linearization of parameter coordinate on master if determined by a
     *         point-to-curve projection
     *
     */
    template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
    void calc_linearization_point_to_curve_projection_parameter_coord_master(
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, T>& lin_xi_master_slaveDofs,
        Core::LinAlg::Matrix<1, 3 * numnodes * numnodalvalues, T>& lin_xi_master_masterDofs,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixi_master,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, double>& N_slave,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, T>& N_master,
        const Core::LinAlg::Matrix<3, 3 * numnodes * numnodalvalues, T>& N_xi_master);

    /** \brief point-to-curve projection:
     *         partial derivatives of the parameter coordinate on master xi_master with respect to
     *         centerline position of slave point, master point and centerline tangent of master
     *
     */
    template <typename T>
    void calc_point_to_curve_projection_parameter_coord_master_partial_derivs(
        Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_slave,
        Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_master,
        Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixi_master);

    /** \brief point-to-curve projection:
     *         partial second derivatives of the parameter coordinate on master xi_master with
     *         respect to centerline position of slave point, master point and centerline tangent of
     *         master
     *
     */
    template <typename T>
    void calc_point_to_curve_projection_parameter_coord_master_partial2nd_derivs(
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_slave,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_xi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_xixi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_slave,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_xi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_xixi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_slave,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_xi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_xixi_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_slave,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_master,
        Core::LinAlg::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_xi_master,
        const Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_slave,
        const Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_master,
        const Core::LinAlg::Matrix<1, 3, T>& xi_master_partial_r_xi_master,
        const Core::LinAlg::Matrix<3, 3, T>& delta_r_deriv_r_slave,
        const Core::LinAlg::Matrix<3, 3, T>& delta_r_deriv_r_master,
        const Core::LinAlg::Matrix<3, 3, T>& delta_r_deriv_r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixixi_master);

    /** \brief point-to-curve projection:
     *         partial derivative of the orthogonality condition with respect to parameter
     * coordinate on master xi_master
     *
     */
    template <typename T>
    void calc_ptc_projection_orthogonality_condition_partial_deriv_parameter_coord_master(
        T& orthogon_condition_partial_xi_master, const Core::LinAlg::Matrix<3, 1, T>& delta_r,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xixi_master);

    /** \brief point-to-curve projection:
     *         partial derivative of the orthogonality condition with respect to centerline position
     *         on slave
     *
     */
    template <typename T>
    void calc_ptc_projection_orthogonality_condition_partial_deriv_cl_pos_slave(
        Core::LinAlg::Matrix<1, 3, T>& orthogon_condition_partial_r_slave,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master);

    /** \brief point-to-curve projection:
     *         partial derivative of the orthogonality condition with respect to centerline position
     *         on master
     *
     */
    template <typename T>
    void calc_ptc_projection_orthogonality_condition_partial_deriv_cl_pos_master(
        Core::LinAlg::Matrix<1, 3, T>& orthogon_condition_partial_r_master,
        const Core::LinAlg::Matrix<3, 1, T>& r_xi_master);

    /** \brief point-to-curve projection:
     *         partial derivative of the orthogonality condition with respect to centerline tangent
     *         on master
     *
     */
    template <typename T>
    void calc_ptc_projection_orthogonality_condition_partial_deriv_cl_tangent_master(
        Core::LinAlg::Matrix<1, 3, T>& orthogon_condition_partial_r_xi_master,
        const Core::LinAlg::Matrix<3, 1, T>& delta_r);

    /** \brief calculate angle enclosed by two vectors a and b
     *
     */
    template <typename T>
    void calc_enclosed_angle(T& angle, T& cosine_angle, const Core::LinAlg::Matrix<3, 1, T>& a,
        const Core::LinAlg::Matrix<3, 1, T>& b);

  }  // namespace Geo
}  // namespace BeamInteraction

FOUR_C_NAMESPACE_CLOSE

#endif

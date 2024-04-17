/*-----------------------------------------------------------------------------------------------*/
/*! \file

\brief utility functions for geometric problems associated with beam-to-? interactions

\level 3

*/
/*-----------------------------------------------------------------------------------------------*/

#include "baci_beaminteraction_geometry_utils.hpp"

#include "baci_beam3_spatial_discretization_utils.hpp"
#include "baci_io_pstream.hpp"
#include "baci_utils_fad.hpp"

#include <Sacado.hpp>

FOUR_C_NAMESPACE_OPEN

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
bool BEAMINTERACTION::GEO::PointToCurveProjection(CORE::LINALG::Matrix<3, 1, T> const& r_slave,
    T& xi_master, double const& xi_master_initial_guess,
    const CORE::LINALG::Matrix<3 * numnodes * numnodalvalues, 1, T>& master_centerline_dof_values,
    const CORE::FE::CellType& master_distype, double master_ele_ref_length)
{
  // vectors for shape functions values and their derivatives
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i_xi(true);
  CORE::LINALG::Matrix<1, numnodes * numnodalvalues, T> N_i_xixi(true);

  // coords and derivatives of the two contacting points
  CORE::LINALG::Matrix<3, 1, T> r_master(true);
  CORE::LINALG::Matrix<3, 1, T> r_xi_master(true);
  CORE::LINALG::Matrix<3, 1, T> r_xixi_master(true);

  CORE::LINALG::Matrix<3, 1, T> delta_r(true);  // = r1-r2

  // initialize function f and Jacobian df for Newton iteration
  T f = 0.0;
  T df = 0.0;

  // scalar residual
  double residual = 0.0;
  double residual0 = 0.0;

  xi_master = xi_master_initial_guess;
  double xi_master_previous_iteration = xi_master_initial_guess;


  // local Newton iteration
  for (unsigned int iter = 0; iter < POINT_TO_CURVE_PROJECTION_MAX_NUM_ITER; ++iter)
  {
    // update shape functions and their derivatives
    DRT::UTILS::BEAM::EvaluateShapeFunctionsAndDerivsAnd2ndDerivsAtXi<numnodes, numnodalvalues, T>(
        xi_master, N_i, N_i_xi, N_i_xixi, master_distype, master_ele_ref_length);

    // update coordinates and derivatives of contact points
    DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3, T, T>(
        master_centerline_dof_values, N_i, r_master);
    DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3, T, T>(
        master_centerline_dof_values, N_i_xi, r_xi_master);
    DRT::UTILS::BEAM::CalcInterpolation<numnodes, numnodalvalues, 3, T, T>(
        master_centerline_dof_values, N_i_xixi, r_xixi_master);

    // use delta_r = r1-r2 as auxiliary quantity
    delta_r = CORE::FADUTILS::DiffVector(r_slave, r_master);

    // compute norm of difference vector to scale the equations
    // (this yields better conditioning)
    /* Note: Even if automatic differentiation via FAD is applied, norm_delta_r has to be of type
     * double since this factor is needed for a pure scaling of the nonlinear CCP and has not to be
     * linearized! */
    double norm_delta_r = CORE::FADUTILS::CastToDouble(CORE::FADUTILS::VectorNorm(delta_r));

    /* The closer the beams get, the smaller is norm_delta_r, but
     * norm_delta_r is not allowed to be too small, else numerical problems occur.
     * It can happen quite often that the centerlines of two beam elements of the same physical beam
     * cross in one point and norm_delta_r = 0. Since in this case |eta1|>1 and |eta2|>1 they will
     * be sorted out later anyways. */
    if (norm_delta_r < POINT_TO_CURVE_PROJECTION_IDENTICAL_POINTS_TOLERANCE)
      dserror("Point-to-curve projection fails because point lies on the curve!");


    // evaluate f at current eta1, eta2
    EvaluatePointToCurveOrthogonalityCondition(f, delta_r, norm_delta_r, r_xi_master);

    // compute the scalar residuum
    /* The residual is scaled with 1/element_length of the beam element representing the curve
     * since r_xi scales with the element_length */
    residual = std::abs(CORE::FADUTILS::CastToDouble(f / master_ele_ref_length));

    // store initial residual
    if (iter == 0) residual0 = residual;

    // check if Newton iteration has converged
    if (CORE::FADUTILS::CastToDouble(residual) < POINT_TO_CURVE_PROJECTION_TOLERANCE_RESIDUUM and
        std::abs(xi_master_previous_iteration - CORE::FADUTILS::CastToDouble(xi_master)) <
            POINT_TO_CURVE_PROJECTION_TOLERANCE_INCREMENT)
    {
      IO::cout(IO::debug) << "\nPoint-to-Curve projection: local Newton loop "
                          << "converged after " << iter << " iterations!" << IO::endl;

      return true;
    }

    // evaluate Jacobian of f at current point
    // Note: It has to be checked, if the linearization is equal to zero;
    bool validlinearization = EvaluateLinearizationPointToCurveOrthogonalityCondition(
        df, delta_r, norm_delta_r, r_xi_master, r_xixi_master);

    if (not validlinearization)
      dserror(
          "Linearization of point to line projection is zero, i.e. the minimal distance "
          "problem seems to be non-unique!");

    xi_master_previous_iteration = CORE::FADUTILS::CastToDouble(xi_master);

    // update master element coordinate of closest point
    xi_master += -f / df;
  }

  // Newton iteration unconverged after maximum number of iterations, print debug info to screen
  IO::cout(IO::debug) << "\n\nWARNING Point-to-Curve projection: local Newton loop "
                      << "unconverged after " << POINT_TO_CURVE_PROJECTION_MAX_NUM_ITER
                      << " iterations!" << IO::endl;
  IO::cout(IO::debug) << "residual in first iteration: " << residual0 << IO::endl;
  IO::cout(IO::debug) << "residual: " << residual << IO::endl;
  IO::cout(IO::debug) << "xi_master: " << CORE::FADUTILS::CastToDouble(xi_master) << IO::endl;
  IO::cout(IO::debug) << "xi_master_previous_iteration: " << xi_master_previous_iteration
                      << IO::endl;

  return false;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::EvaluatePointToCurveOrthogonalityCondition(T& f,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r, const double norm_delta_r,
    const CORE::LINALG::Matrix<3, 1, T>& r_xi_master)
{
  // reset f
  f = 0.0;

  // evaluate f
  for (unsigned int i = 0; i < 3; ++i)
  {
    f += -delta_r(i) * r_xi_master(i) / norm_delta_r;
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
bool BEAMINTERACTION::GEO::EvaluateLinearizationPointToCurveOrthogonalityCondition(T& df,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r, const double norm_delta_r,
    const CORE::LINALG::Matrix<3, 1, T>& r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixi_master)
{
  // reset df
  df = 0.0;

  // evaluate df
  for (unsigned int i = 0; i < 3; ++i)
  {
    df += (r_xi_master(i) * r_xi_master(i) - delta_r(i) * r_xixi_master(i)) / norm_delta_r;
  }

  /* check for df==0.0, i.e. non-uniqueness of the minimal distance problem
   * This can happen e.g. when the curve describes a circle geometry and
   * the projecting slave point coincides with the center of the circle */
  if (std::abs(CORE::FADUTILS::CastToDouble(df)) <
      POINT_TO_CURVE_PROJECTION_NONUNIQUE_MINIMAL_DISTANCE_TOLERANCE)
    return false;
  else
    return true;
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <unsigned int numnodes, unsigned int numnodalvalues, typename T>
void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster(
    CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, T>& lin_xi_master_slaveDofs,
    CORE::LINALG::Matrix<1, 3 * numnodes * numnodalvalues, T>& lin_xi_master_masterDofs,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r, const CORE::LINALG::Matrix<3, 1, T>& r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixi_master,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, double>& N_slave,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, T>& N_master,
    const CORE::LINALG::Matrix<3, 3 * numnodes * numnodalvalues, T>& N_xi_master)
{
  const unsigned int dim1 = 3 * numnodes * numnodalvalues;
  const unsigned int dim2 = 3 * numnodes * numnodalvalues;

  /* partial derivative of the orthogonality condition with respect to parameter coordinate on
   * master xi_master */
  T orthogon_condition_partial_xi_master = 0.0;

  CalcPTCProjectionOrthogonalityConditionPartialDerivParameterCoordMaster(
      orthogon_condition_partial_xi_master, delta_r, r_xi_master, r_xixi_master);

  /* partial derivative of the orthogonality condition with respect to primary Dofs defining
   * slave point and master curve */
  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_slave(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClPosSlave(
      orthogon_condition_partial_r_slave, r_xi_master);

  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_master(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClPosMaster(
      orthogon_condition_partial_r_master, r_xi_master);

  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_xi_master(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClTangentMaster(
      orthogon_condition_partial_r_xi_master, delta_r);


  // finally compute the linearizations / directional derivatives
  lin_xi_master_slaveDofs.Clear();
  lin_xi_master_masterDofs.Clear();

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdof = 0; jdof < dim1; ++jdof)
    {
      lin_xi_master_slaveDofs(jdof) +=
          orthogon_condition_partial_r_slave(idim) * N_slave(idim, jdof);
    }

  for (unsigned int idim = 0; idim < 3; ++idim)
    for (unsigned int jdof = 0; jdof < dim2; ++jdof)
    {
      lin_xi_master_masterDofs(jdof) +=
          orthogon_condition_partial_r_master(idim) * N_master(idim, jdof) +
          orthogon_condition_partial_r_xi_master(idim) * N_xi_master(idim, jdof);
    }

  lin_xi_master_slaveDofs.Scale(-1.0 / orthogon_condition_partial_xi_master);
  lin_xi_master_masterDofs.Scale(-1.0 / orthogon_condition_partial_xi_master);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartialDerivs(
    CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_slave,
    CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_master,
    CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r, const CORE::LINALG::Matrix<3, 1, T>& r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixi_master)
{
  /* partial derivative of the orthogonality condition with respect to parameter coordinate on
   * master xi_master */
  T orthogon_condition_partial_xi_master = 0.0;

  CalcPTCProjectionOrthogonalityConditionPartialDerivParameterCoordMaster(
      orthogon_condition_partial_xi_master, delta_r, r_xi_master, r_xixi_master);

  /* partial derivative of the orthogonality condition with respect to primary Dofs defining
   * slave point and master curve */
  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_slave(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClPosSlave(
      orthogon_condition_partial_r_slave, r_xi_master);

  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_master(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClPosMaster(
      orthogon_condition_partial_r_master, r_xi_master);

  CORE::LINALG::Matrix<1, 3, T> orthogon_condition_partial_r_xi_master(true);

  CalcPTCProjectionOrthogonalityConditionPartialDerivClTangentMaster(
      orthogon_condition_partial_r_xi_master, delta_r);

  // finally compute the partial/directional derivatives
  xi_master_partial_r_slave.Update(
      -1.0 / orthogon_condition_partial_xi_master, orthogon_condition_partial_r_slave);

  xi_master_partial_r_master.Update(
      -1.0 / orthogon_condition_partial_xi_master, orthogon_condition_partial_r_master);

  xi_master_partial_r_xi_master.Update(
      -1.0 / orthogon_condition_partial_xi_master, orthogon_condition_partial_r_xi_master);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartial2ndDerivs(
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_slave,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_xi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_slave_partial_r_xixi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_slave,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_xi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_master_partial_r_xixi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_slave,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_xi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xi_master_partial_r_xixi_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_slave,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_master,
    CORE::LINALG::Matrix<3, 3, T>& xi_master_partial_r_xixi_master_partial_r_xi_master,
    const CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_slave,
    const CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_master,
    const CORE::LINALG::Matrix<1, 3, T>& xi_master_partial_r_xi_master,
    const CORE::LINALG::Matrix<3, 3, T>& delta_r_deriv_r_slave,
    const CORE::LINALG::Matrix<3, 3, T>& delta_r_deriv_r_master,
    const CORE::LINALG::Matrix<3, 3, T>& delta_r_deriv_r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r, const CORE::LINALG::Matrix<3, 1, T>& r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixixi_master)
{
  //  Fixme: the (out) variables should be called (...)_partial_(...)_deriv_(...) !

  /* partial derivative of the orthogonality condition with respect to parameter coordinate on
   * master xi_master */
  T orthogon_condition_partial_xi_master = 0.0;

  CalcPTCProjectionOrthogonalityConditionPartialDerivParameterCoordMaster(
      orthogon_condition_partial_xi_master, delta_r, r_xi_master, r_xixi_master);

  T orthogon_condition_partial_xi_master_inverse = 1.0 / orthogon_condition_partial_xi_master;

  // Note: 1) do (partial) derivs w.r.t. [r_master(xi_master_c)], [r_xi_master(xi_master_c)] and
  //          [r_xixi_master(xi_master_c)],
  // 2) add the contributions from xi_master_partial_(...) according to the chain rule,
  // 3) add the terms including delta_r_deriv_(...) since these already
  //    contain the contributions from xi_master_partial_(...) according to the chain rule
  // 4) add the contributions from linearization of (variation of r_master) and linearization of
  //    (variation of r_xi_master) according to the chain rule


  CORE::LINALG::Matrix<3, 3, T> unit_matrix(true);
  for (unsigned int i = 0; i < 3; ++i) unit_matrix(i, i) = 1.0;

  CORE::LINALG::Matrix<3, 3, T> r_xi_master_tensorproduct_r_xi_master(true);
  CORE::LINALG::Matrix<3, 3, T> r_xi_master_tensorproduct_r_xixi_master(true);
  CORE::LINALG::Matrix<3, 3, T> r_xi_master_tensorproduct_delta_r(true);
  CORE::LINALG::Matrix<3, 3, T> delta_r_tensorproduct_r_xixi_master(true);
  CORE::LINALG::Matrix<3, 3, T> delta_r_tensorproduct_delta_r(true);

  for (unsigned int irow = 0; irow < 3; ++irow)
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      r_xi_master_tensorproduct_r_xi_master(irow, icol) = r_xi_master(irow) * r_xi_master(icol);
      r_xi_master_tensorproduct_r_xixi_master(irow, icol) = r_xi_master(irow) * r_xixi_master(icol);
      r_xi_master_tensorproduct_delta_r(irow, icol) = r_xi_master(irow) * delta_r(icol);
      delta_r_tensorproduct_r_xixi_master(irow, icol) = delta_r(irow) * r_xixi_master(icol);
      delta_r_tensorproduct_delta_r(irow, icol) = delta_r(irow) * delta_r(icol);
    }

  // 1)
  xi_master_partial_r_slave_partial_r_xi_master.Update(
      -2.0 * orthogon_condition_partial_xi_master_inverse *
          orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xi_master);

  xi_master_partial_r_slave_partial_r_xi_master.Update(
      -1.0 * orthogon_condition_partial_xi_master_inverse, unit_matrix, 1.0);

  xi_master_partial_r_master_partial_r_xi_master.Update(
      -1.0, xi_master_partial_r_slave_partial_r_xi_master);


  xi_master_partial_r_slave_partial_r_xixi_master.Update(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_delta_r);

  xi_master_partial_r_master_partial_r_xixi_master.Update(
      -1.0, xi_master_partial_r_slave_partial_r_xixi_master);


  xi_master_partial_r_xi_master_partial_r_xi_master.UpdateT(
      -2.0 * orthogon_condition_partial_xi_master_inverse *
          orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_delta_r);

  xi_master_partial_r_xi_master_partial_r_xixi_master.Update(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      delta_r_tensorproduct_delta_r);


  // 2)
  // add contributions from linearization of master parameter coordinate xi_master
  // to [.]_deriv_r_xi_master expressions (according to chain rule)
  CORE::LINALG::Matrix<3, 1, T> tmp_vec2;
  tmp_vec2.Multiply(xi_master_partial_r_slave_partial_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_slave_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_slave_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_slave_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(xi_master_partial_r_master_partial_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_master_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_master_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_master_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(xi_master_partial_r_xi_master_partial_r_xi_master, r_xixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_xi_master_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_xi_master_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_xi_master_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // add contributions from linearization of master parameter coordinate xi_master
  // to [.]_deriv_r_xixi_master expressions (according to chain rule)
  tmp_vec2.Multiply(xi_master_partial_r_slave_partial_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_slave_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_slave_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_slave_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(xi_master_partial_r_master_partial_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_master_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_master_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_master_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }

  tmp_vec2.Multiply(xi_master_partial_r_xi_master_partial_r_xixi_master, r_xixixi_master);

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_xi_master_partial_r_slave(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_slave(icol);

      xi_master_partial_r_xi_master_partial_r_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_master(icol);

      xi_master_partial_r_xi_master_partial_r_xi_master(irow, icol) +=
          tmp_vec2(irow) * xi_master_partial_r_xi_master(icol);
    }
  }


  // 3)
  xi_master_partial_r_xi_master_partial_r_slave.Update(
      -1.0 * orthogon_condition_partial_xi_master_inverse, delta_r_deriv_r_slave, 1.0);

  xi_master_partial_r_xi_master_partial_r_master.Update(
      -1.0 * orthogon_condition_partial_xi_master_inverse, delta_r_deriv_r_master, 1.0);

  xi_master_partial_r_xi_master_partial_r_xi_master.Update(
      -1.0 * orthogon_condition_partial_xi_master_inverse, delta_r_deriv_r_xi_master, 1.0);


  xi_master_partial_r_slave_partial_r_slave.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_slave, 1.0);

  xi_master_partial_r_slave_partial_r_master.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_master, 1.0);

  xi_master_partial_r_slave_partial_r_xi_master.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_xi_master, 1.0);


  xi_master_partial_r_master_partial_r_slave.Multiply(
      -1.0 * orthogon_condition_partial_xi_master_inverse *
          orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_slave, 1.0);

  xi_master_partial_r_master_partial_r_master.Multiply(
      -1.0 * orthogon_condition_partial_xi_master_inverse *
          orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_master, 1.0);

  xi_master_partial_r_master_partial_r_xi_master.Multiply(
      -1.0 * orthogon_condition_partial_xi_master_inverse *
          orthogon_condition_partial_xi_master_inverse,
      r_xi_master_tensorproduct_r_xixi_master, delta_r_deriv_r_xi_master, 1.0);


  xi_master_partial_r_xi_master_partial_r_slave.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      delta_r_tensorproduct_r_xixi_master, delta_r_deriv_r_slave, 1.0);

  xi_master_partial_r_xi_master_partial_r_master.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      delta_r_tensorproduct_r_xixi_master, delta_r_deriv_r_master, 1.0);

  xi_master_partial_r_xi_master_partial_r_xi_master.Multiply(
      orthogon_condition_partial_xi_master_inverse * orthogon_condition_partial_xi_master_inverse,
      delta_r_tensorproduct_r_xixi_master, delta_r_deriv_r_xi_master, 1.0);


  // 4)
  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_xi_master_partial_r_slave(irow, icol) +=
          orthogon_condition_partial_xi_master_inverse * r_xi_master(irow) *
          xi_master_partial_r_slave(icol);

      xi_master_partial_r_xi_master_partial_r_master(irow, icol) +=
          orthogon_condition_partial_xi_master_inverse * r_xi_master(irow) *
          xi_master_partial_r_master(icol);

      xi_master_partial_r_xi_master_partial_r_xi_master(irow, icol) +=
          orthogon_condition_partial_xi_master_inverse * r_xi_master(irow) *
          xi_master_partial_r_xi_master(icol);
    }
  }

  for (unsigned int irow = 0; irow < 3; ++irow)
  {
    for (unsigned int icol = 0; icol < 3; ++icol)
    {
      xi_master_partial_r_xixi_master_partial_r_slave(irow, icol) -=
          orthogon_condition_partial_xi_master_inverse * delta_r(irow) *
          xi_master_partial_r_slave(icol);

      xi_master_partial_r_xixi_master_partial_r_master(irow, icol) -=
          orthogon_condition_partial_xi_master_inverse * delta_r(irow) *
          xi_master_partial_r_master(icol);

      xi_master_partial_r_xixi_master_partial_r_xi_master(irow, icol) -=
          orthogon_condition_partial_xi_master_inverse * delta_r(irow) *
          xi_master_partial_r_xi_master(icol);
    }
  }
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPTCProjectionOrthogonalityConditionPartialDerivParameterCoordMaster(
    T& orthogon_condition_partial_xi_master, const CORE::LINALG::Matrix<3, 1, T>& delta_r,
    const CORE::LINALG::Matrix<3, 1, T>& r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xixi_master)
{
  orthogon_condition_partial_xi_master = -r_xi_master.Dot(r_xi_master) + delta_r.Dot(r_xixi_master);

  if (std::abs(CORE::FADUTILS::CastToDouble(orthogon_condition_partial_xi_master)) <
      POINT_TO_CURVE_PROJECTION_NONUNIQUE_MINIMAL_DISTANCE_TOLERANCE)
    dserror(
        "Linearization of point to line projection is zero, i.e. the minimal distance "
        "problem is non-unique!");
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPTCProjectionOrthogonalityConditionPartialDerivClPosSlave(
    CORE::LINALG::Matrix<1, 3, T>& orthogon_condition_partial_r_slave,
    const CORE::LINALG::Matrix<3, 1, T>& r_xi_master)
{
  orthogon_condition_partial_r_slave.UpdateT(r_xi_master);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPTCProjectionOrthogonalityConditionPartialDerivClPosMaster(
    CORE::LINALG::Matrix<1, 3, T>& orthogon_condition_partial_r_master,
    const CORE::LINALG::Matrix<3, 1, T>& r_xi_master)
{
  orthogon_condition_partial_r_master.UpdateT(-1.0, r_xi_master);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcPTCProjectionOrthogonalityConditionPartialDerivClTangentMaster(
    CORE::LINALG::Matrix<1, 3, T>& orthogon_condition_partial_r_xi_master,
    const CORE::LINALG::Matrix<3, 1, T>& delta_r)
{
  orthogon_condition_partial_r_xi_master.UpdateT(delta_r);
}

/*-----------------------------------------------------------------------------------------------*
 *-----------------------------------------------------------------------------------------------*/
template <typename T>
void BEAMINTERACTION::GEO::CalcEnclosedAngle(T& angle, T& cosine_angle,
    const CORE::LINALG::Matrix<3, 1, T>& a, const CORE::LINALG::Matrix<3, 1, T>& b)
{
  if (CORE::FADUTILS::VectorNorm(a) < 1.0e-12 or CORE::FADUTILS::VectorNorm(b) < 1.0e-12)
    dserror("Cannot determine angle for zero vector!");

  cosine_angle = CORE::FADUTILS::Norm<T>(
      a.Dot(b) / (CORE::FADUTILS::VectorNorm(a) * CORE::FADUTILS::VectorNorm(b)));

  if (cosine_angle < 1.0)
  {
    // returns an angle \in [0;pi/2] since scalarproduct \in [0;1.0]
    angle = std::acos(cosine_angle);
  }
  else
  {
    /* This step is necessary due to round-off errors.
     * However, the derivative information of the FAD quantity gets lost here! */
    angle = 0.0;
  }

  // We want an angle \in [0;pi/2] in each case:
  if (angle > M_PI / 2.0)
    dserror("Something went wrong here, angle should be in the interval [0;pi/2]!");
}


// explicit template instantiations
template bool BEAMINTERACTION::GEO::PointToCurveProjection<2, 1, double>(
    CORE::LINALG::Matrix<3, 1, double> const&, double&, double const&,
    const CORE::LINALG::Matrix<6, 1, double>&, const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<3, 1, double>(
    CORE::LINALG::Matrix<3, 1, double> const&, double&, double const&,
    const CORE::LINALG::Matrix<9, 1, double>&, const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<4, 1, double>(
    CORE::LINALG::Matrix<3, 1, double> const&, double&, double const&,
    const CORE::LINALG::Matrix<12, 1, double>&, const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<5, 1, double>(
    CORE::LINALG::Matrix<3, 1, double> const&, double&, double const&,
    const CORE::LINALG::Matrix<15, 1, double>&, const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<2, 2, double>(
    CORE::LINALG::Matrix<3, 1, double> const&, double&, double const&,
    const CORE::LINALG::Matrix<12, 1, double>&, const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<2, 1, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>> const&, Sacado::Fad::DFad<double>&,
    double const&, const CORE::LINALG::Matrix<6, 1, Sacado::Fad::DFad<double>>&,
    const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<3, 1, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>> const&, Sacado::Fad::DFad<double>&,
    double const&, const CORE::LINALG::Matrix<9, 1, Sacado::Fad::DFad<double>>&,
    const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<4, 1, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>> const&, Sacado::Fad::DFad<double>&,
    double const&, const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<5, 1, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>> const&, Sacado::Fad::DFad<double>&,
    double const&, const CORE::LINALG::Matrix<15, 1, Sacado::Fad::DFad<double>>&,
    const CORE::FE::CellType&, double);
template bool BEAMINTERACTION::GEO::PointToCurveProjection<2, 2, Sacado::Fad::DFad<double>>(
    CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>> const&, Sacado::Fad::DFad<double>&,
    double const&, const CORE::LINALG::Matrix<12, 1, Sacado::Fad::DFad<double>>&,
    const CORE::FE::CellType&, double);

template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<2,
    1, double>(CORE::LINALG::Matrix<1, 6, double>&, CORE::LINALG::Matrix<1, 6, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 6, double>&,
    const CORE::LINALG::Matrix<3, 6, double>&, const CORE::LINALG::Matrix<3, 6, double>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<3,
    1, double>(CORE::LINALG::Matrix<1, 9, double>&, CORE::LINALG::Matrix<1, 9, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 9, double>&,
    const CORE::LINALG::Matrix<3, 9, double>&, const CORE::LINALG::Matrix<3, 9, double>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<4,
    1, double>(CORE::LINALG::Matrix<1, 12, double>&, CORE::LINALG::Matrix<1, 12, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 12, double>&,
    const CORE::LINALG::Matrix<3, 12, double>&, const CORE::LINALG::Matrix<3, 12, double>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<5,
    1, double>(CORE::LINALG::Matrix<1, 15, double>&, CORE::LINALG::Matrix<1, 15, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 15, double>&,
    const CORE::LINALG::Matrix<3, 15, double>&, const CORE::LINALG::Matrix<3, 15, double>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<2,
    2, double>(CORE::LINALG::Matrix<1, 12, double>&, CORE::LINALG::Matrix<1, 12, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 12, double>&,
    const CORE::LINALG::Matrix<3, 12, double>&, const CORE::LINALG::Matrix<3, 12, double>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<2,
    1, Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 6, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 6, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 6, double>&,
    const CORE::LINALG::Matrix<3, 6, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 6, Sacado::Fad::DFad<double>>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<3,
    1, Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 9, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 9, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 9, double>&,
    const CORE::LINALG::Matrix<3, 9, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 9, Sacado::Fad::DFad<double>>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<4,
    1, Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 12, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 12, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 12, double>&,
    const CORE::LINALG::Matrix<3, 12, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 12, Sacado::Fad::DFad<double>>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<5,
    1, Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 15, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 15, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 15, double>&,
    const CORE::LINALG::Matrix<3, 15, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 15, Sacado::Fad::DFad<double>>&);
template void BEAMINTERACTION::GEO::CalcLinearizationPointToCurveProjectionParameterCoordMaster<2,
    2, Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 12, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 12, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 12, double>&,
    const CORE::LINALG::Matrix<3, 12, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 12, Sacado::Fad::DFad<double>>&);

template void
BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartialDerivs<double>(
    CORE::LINALG::Matrix<1, 3, double>&, CORE::LINALG::Matrix<1, 3, double>&,
    CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&);
template void BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartialDerivs<
    Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&);

template void
BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartial2ndDerivs<double>(
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, CORE::LINALG::Matrix<3, 3, double>&,
    CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<1, 3, double>&,
    const CORE::LINALG::Matrix<1, 3, double>&, const CORE::LINALG::Matrix<1, 3, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 3, double>&,
    const CORE::LINALG::Matrix<3, 3, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&,
    const CORE::LINALG::Matrix<3, 1, double>&);
template void BEAMINTERACTION::GEO::CalcPointToCurveProjectionParameterCoordMasterPartial2ndDerivs<
    Sacado::Fad::DFad<double>>(CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<1, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 3, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&);

template void BEAMINTERACTION::GEO::CalcEnclosedAngle<double>(double&, double&,
    const CORE::LINALG::Matrix<3, 1, double>&, const CORE::LINALG::Matrix<3, 1, double>&);
template void BEAMINTERACTION::GEO::CalcEnclosedAngle<Sacado::Fad::DFad<double>>(
    Sacado::Fad::DFad<double>&, Sacado::Fad::DFad<double>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&,
    const CORE::LINALG::Matrix<3, 1, Sacado::Fad::DFad<double>>&);

FOUR_C_NAMESPACE_CLOSE

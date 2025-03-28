// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_fem_geometry_position_array.hpp"
#include "4C_fem_nurbs_discretization_utils.hpp"
#include "4C_scatra_ele.hpp"
#include "4C_scatra_ele_action.hpp"
#include "4C_scatra_ele_calc.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 * Action type: Evaluate
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::evaluate_od(Core::Elements::Element* ele,
    Teuchos::ParameterList& params, Core::FE::Discretization& discretization,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  //--------------------------------------------------------------------------------
  // preparations for element
  //--------------------------------------------------------------------------------

  if (setup_calc(ele, discretization) == -1) return 0;

  //--------------------------------------------------------------------------------
  // extract element based or nodal values
  //--------------------------------------------------------------------------------

  extract_element_and_node_values(ele, params, discretization, la);

  //--------------------------------------------------------------------------------
  // calculate element coefficient matrix
  //--------------------------------------------------------------------------------

  // extract action parameter
  const auto action = Teuchos::getIntegralValue<ScaTra::Action>(params, "action");

  // evaluate action
  evaluate_action_od(ele, params, discretization, action, la, elemat1_epetra, elemat2_epetra,
      elevec1_epetra, elevec2_epetra, elevec3_epetra);

  return 0;
}


/*----------------------------------------------------------------------*
 | evaluate action for off-diagonal system matrix block      fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
int Discret::Elements::ScaTraEleCalc<distype, probdim>::evaluate_action_od(
    Core::Elements::Element* ele, Teuchos::ParameterList& params,
    Core::FE::Discretization& discretization, const ScaTra::Action& action,
    Core::Elements::LocationArray& la, Core::LinAlg::SerialDenseMatrix& elemat1_epetra,
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,
    Core::LinAlg::SerialDenseVector& elevec1_epetra,
    Core::LinAlg::SerialDenseVector& elevec2_epetra,
    Core::LinAlg::SerialDenseVector& elevec3_epetra)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_scatra_mono_odblock_mesh:
    {
      sysmat_od_mesh(ele, elemat1_epetra, nsd_);

      break;
    }

    case ScaTra::Action::calc_scatra_mono_odblock_fluid:
    {
      sysmat_od_fluid(ele, elemat1_epetra, nsd_ + 1);

      break;
    }

    default:
    {
      FOUR_C_THROW("Not acting on action {}. Forgot implementation?", action);
      break;
    }
  }  // switch(action)

  return 0;
}


/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 vuong 08/14|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::sysmat_od_mesh(
    Core::Elements::Element* ele,           ///< the element those matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    const int ndofpernodemesh               ///< number of DOF of mesh displacement field
)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // const double vol=eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // get material and stabilization parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // material parameter at the element center are also necessary
  // even if the stabilization parameter is evaluated at the element center
  if (not scatrapara_->mat_gp())
  {
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  //----------------------------------------------------------------------
  // calculation of subgrid diffusivity and stabilization parameter(s)
  // at element center
  //----------------------------------------------------------------------

  // the stabilization parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    // velocity divergence required for conservative form
    double vdiv(0.0);
    if (scatrapara_->is_conservative()) get_divergence(vdiv, evelnp_);

    //------------------------------------------------dJ/dd = dJ/dF : dF/dd = J * F^-T . N_{\psi} =
    // J * N_x
    // J denotes the determinant of the Jacobian of the mapping between current and parameter space,
    // i.e. det(dx/ds)
    static Core::LinAlg::Matrix<1, nsd_ * nen_> dJ_dmesh(
        Core::LinAlg::Initialization::uninitialized);
    calc_djd_mesh(dJ_dmesh);
    const double J = xjm_.determinant();

    // loop all scalars
    for (int k = 0; k < numscal_; ++k)  // deal with a system of transported scalars
    {
      // reactive part of the form: (reaction coefficient)*phi
      double rea_phi(0.0);
      rea_phi = densnp[k] * scatravarmanager_->phinp(k) * reamanager_->get_rea_coeff(k);

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term
      double rhsint(0.0);
      get_rhs_int(rhsint, densnp[k], k);


      // subgrid-scale convective term
      Core::LinAlg::Matrix<nen_, 1> sgconv(Core::LinAlg::Initialization::zero);
      // subgrid-scale velocity vector in gausspoint
      Core::LinAlg::Matrix<nsd_, 1> sgvelint(Core::LinAlg::Initialization::zero);

      // residual of convection-diffusion-reaction eq
      double scatrares(0.0);

      // compute residual of scalar transport equation and
      // subgrid-scale part of scalar
      calc_strong_residual(k, scatrares, densam[k], densnp[k], rea_phi, rhsint, tau[k]);

      double rhsfac = scatraparatimint_->time_fac_rhs() * fac;

      //----------------------------------------------------------------
      // standard Galerkin terms
      //----------------------------------------------------------------

      if (scatraparatimint_->is_incremental() and not scatraparatimint_->is_stationary())
        calc_lin_mass_od_mesh(emat, k, ndofpernodemesh, rhsfac, fac, densam[k], densnp[k],
            scatravarmanager_->phinp(k), scatravarmanager_->hist(k), J, dJ_dmesh);

      // the order of the following three functions is important
      // and must not be changed
      compute_rhs_int(rhsint, densam[k], densnp[k], scatravarmanager_->hist(k));

      // diffusive part used in stabilization terms
      Core::LinAlg::Matrix<nen_, 1> diff(Core::LinAlg::Initialization::zero);
      // diffusive term using current scalar value for higher-order elements
      if (use2ndderiv_)
      {
        // diffusive part:  diffus * ( N,xx  +  N,yy +  N,zz )
        get_laplacian_strong_form(diff);
        diff.scale(diffmanager_->get_isotropic_diff(k));
      }

      recompute_scatra_res_for_rhs(scatrares, k, diff, densn[k], densnp[k], rea_phi, rhsint);

      recompute_conv_phi_for_rhs(k, sgvelint, densnp[k], densn[k], vdiv);

      //----------------------------------------------------------------
      // standard Galerkin transient, old part of rhs and bodyforce term
      //----------------------------------------------------------------
      calc_hist_and_source_od_mesh(emat, k, ndofpernodemesh, fac, rhsint, J, dJ_dmesh, densnp[k]);

      //----------------------------------------------------------------
      // standard Galerkin terms - convective term
      //----------------------------------------------------------------
      if (not scatrapara_->is_conservative())
      {
        calc_conv_od_mesh(emat, k, ndofpernodemesh, fac, rhsfac, densnp[k], J,
            scatravarmanager_->grad_phi(k), scatravarmanager_->con_vel(k));
      }
      else
        calc_conv_cons_od_mesh(emat, k, fac, scatraparatimint_->time_fac() * fac, densnp[k], J);

      //----------------------------------------------------------------
      // standard Galerkin terms  --  diffusive term
      //----------------------------------------------------------------
      calc_diff_od_mesh(emat, k, ndofpernodemesh, diffmanager_->get_isotropic_diff(k), fac, rhsfac,
          J, scatravarmanager_->grad_phi(k), scatravarmanager_->con_vel(k), dJ_dmesh);

      //----------------------------------------------------------------
      // standard Galerkin terms  -- "shapederivatives" reactive term
      //----------------------------------------------------------------
      calc_react_od_mesh(emat, k, ndofpernodemesh, rhsfac, rea_phi, J, dJ_dmesh);

    }  // end loop all scalars

  }  // end loop Gauss points
}

/*----------------------------------------------------------------------*
|  calculate system matrix and rhs (public)                 vuong 08/14|
*----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::sysmat_od_fluid(
    Core::Elements::Element* ele,           ///< the element those matrix is calculated
    Core::LinAlg::SerialDenseMatrix& emat,  ///< element matrix to calculate
    const int numdofpernode_fluid           ///< number of DOF of fluid field
)
{
  //----------------------------------------------------------------------
  // calculation of element volume both for tau at ele. cent. and int. pt.
  //----------------------------------------------------------------------
  // const double vol=eval_shape_func_and_derivs_at_ele_center();

  //----------------------------------------------------------------------
  // get material and stabilization parameters (evaluation at element center)
  //----------------------------------------------------------------------
  // density at t_(n)
  std::vector<double> densn(numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(numscal_, 1.0);

  // fluid viscosity
  double visc(0.0);

  // material parameter at the element center are also necessary
  // even if the stabilization parameter is evaluated at the element center
  if (not scatrapara_->mat_gp())
  {
    set_internal_variables_for_mat_and_rhs();

    get_material_params(ele, densn, densnp, densam, visc);
  }

  //----------------------------------------------------------------------
  // calculation of subgrid diffusivity and stabilization parameter(s)
  // at element center
  //----------------------------------------------------------------------

  // the stabilization parameters (one per transported scalar)
  std::vector<double> tau(numscal_, 0.0);

  //----------------------------------------------------------------------
  // integration loop for one element
  //----------------------------------------------------------------------
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    const double fac = eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    set_internal_variables_for_mat_and_rhs();

    //----------------------------------------------------------------------
    // get material parameters (evaluation at integration point)
    //----------------------------------------------------------------------
    if (scatrapara_->mat_gp()) get_material_params(ele, densn, densnp, densam, visc, iquad);

    // velocity divergence required for conservative form
    double vdiv(0.0);
    if (scatrapara_->is_conservative()) get_divergence(vdiv, evelnp_);

    // loop all scalars
    for (int k = 0; k < numscal_; ++k)  // deal with a system of transported scalars
    {
      // reactive part of the form: (reaction coefficient)*phi
      double rea_phi(0.0);
      rea_phi = densnp[k] * scatravarmanager_->phinp(k) * reamanager_->get_rea_coeff(k);

      // compute rhs containing bodyforce (divided by specific heat capacity) and,
      // for temperature equation, the time derivative of thermodynamic pressure,
      // if not constant, and for temperature equation of a reactive
      // equation system, the reaction-rate term
      double rhsint(0.0);
      get_rhs_int(rhsint, densnp[k], k);

      const double timefacfac = scatraparatimint_->time_fac() * fac;

      //----------------------------------------------------------------
      // standard Galerkin terms
      //----------------------------------------------------------------

      if (scatraparatimint_->is_incremental() and not scatraparatimint_->is_stationary())
        calc_lin_mass_od_fluid(emat, k, numdofpernode_fluid, timefacfac, fac, densam[k], densnp[k],
            scatravarmanager_->phinp(k), scatravarmanager_->hist(k));

      compute_rhs_int(rhsint, densam[k], densnp[k], scatravarmanager_->hist(k));

      //----------------------------------------------------------------
      // standard Galerkin transient, old part of rhs and bodyforce term
      //----------------------------------------------------------------
      calc_hist_and_source_od_fluid(emat, k, numdofpernode_fluid, fac, rhsint, densnp[k]);

      //----------------------------------------------------------------
      // standard Galerkin terms - convective term
      //----------------------------------------------------------------
      // calculation of convective element matrix in convective form
      calc_mat_conv_od_fluid(
          emat, k, numdofpernode_fluid, timefacfac, densnp[k], scatravarmanager_->grad_phi(k));

      // add conservative contributions
      if (scatrapara_->is_conservative())
        calc_mat_conv_add_cons_od_fluid(
            emat, k, numdofpernode_fluid, timefacfac, densnp[k], scatravarmanager_->phinp(k));

      //----------------------------------------------------------------
      // standard Galerkin terms  --  diffusive term
      //----------------------------------------------------------------
      calc_diff_od_fluid(emat, k, numdofpernode_fluid, timefacfac, scatravarmanager_->grad_phi(k));

      //----------------------------------------------------------------
      // standard Galerkin terms  -- "shapederivatives" reactive term
      //----------------------------------------------------------------
      calc_react_od_fluid(emat, k, numdofpernode_fluid, timefacfac, rea_phi);

    }  // end loop all scalars

  }  // end loop Gauss points
}


/*-----------------------------------------------------------------------------*
 |  calculation of convective element matrix                                    |
 |  in convective form (OD fluid)                                   vuong 08/14 |
 *-----------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_conv_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodefluid,
    const double timefacfac, const double densnp, const Core::LinAlg::Matrix<nsd_, 1>& gradphi)
{
  // convective term in convective form
  const double densfac = timefacfac * densnp;
  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    const double v = densfac * funct_(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const int fui = ui * ndofpernodefluid;

      for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
        emat(fvi, fui + udim) += v * funct_(ui) * gradphi(udim);
    }
  }
}

/*-----------------------------------------------------------------------------*
 |   calculation of convective element matrix: add conservative                 |
 |   contributions (OD fluid)                                       vuong 08/14 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_mat_conv_add_cons_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodefluid,
    const double timefacfac, const double densnp, const double phinp)
{
  const double consfac = timefacfac * densnp * phinp;
  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    const double v = consfac * funct_(vi);
    const int fvi = vi * numdofpernode_ + k;

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const int fui = ui * ndofpernodefluid;

      for (unsigned udim = 0; udim < nsd_; ++udim) emat(fvi, fui) += v * derxy_(udim, udim);
    }
  }
}

/*------------------------------------------------------------------- *
 |  calculation of linearized mass (OD mesh)             vuong 08/14 |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_lin_mass_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double fac, const double densam, const double densnp,
    const double phinp, const double hist, const double J,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  double vtrans = 0.0;

  if (scatraparatimint_->is_gen_alpha())
    vtrans = rhsfac * densam * hist / J;
  else
  {
    // compute scalar at integration point
    vtrans = fac * densnp * phinp / J;
  }

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;

    const double val = vtrans * funct_(vi);
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const int fui = ui * ndofpernodemesh;

      for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
        emat(fvi, fui + udim) += val * dJ_dmesh(fui + udim);
    }
  }
}

/*-------------------------------------------------------------------------------------- *
 |  standard Galerkin transient, old part of rhs and source term  (OD mesh)   vuong 08/14 |
 *---------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_hist_and_source_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsint, const double J, const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh,
    const double densnp)
{
  double vrhs = -1.0 * fac / J * rhsint;
  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    const int fvi = vi * numdofpernode_ + k;
    const double val = vrhs * funct_(vi);

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const int fui = ui * ndofpernodemesh;

      for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
        emat(fvi, fui + udim) += val * dJ_dmesh(fui + udim);
    }
  }
}

/*-------------------------------------------------------------------- *
 |  standard Galerkin convective term (OD mesh)            vuong 08/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_conv_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsfac, const double densnp, const double J,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi, const Core::LinAlg::Matrix<nsd_, 1>& convelint)
{
  if (not scatraparatimint_->is_stationary())
  {
    // convective term in convective form
    const double densfac = fac * densnp;
    for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    {
      const double v = densfac * funct_(vi);
      const int fvi = vi * numdofpernode_ + k;

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const int fui = ui * ndofpernodemesh;

        for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
          emat(fvi, fui + udim) += -1.0 * v * funct_(ui) * gradphi(udim);
      }
    }
  }

  //----------------------------------------------------------------
  // standard Galerkin terms  -- "shapederivatives" convective term
  //----------------------------------------------------------------
  apply_shape_derivs_conv(emat, k, rhsfac, densnp, J, gradphi, convelint);
}


/*------------------------------------------------------------------------------ *
 | standard Galerkin convective term in conservative form (OD mesh)   fang 08/17 |
 *-------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_conv_cons_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const int k,                            //!< species index
    const double fac,                       //!< domain-integration factor
    const double timefacfac,  //!< time-integration factor times domain-integration factor
    const double densnp,      //!< density
    const double J  //!< Jacobian determinant of mapping between spatial and parameter coordinates
) const
{
  if (not scatraparatimint_->is_stationary())
  {
    // d (phi div v_s)
    // --------------- = phi*derxy*(d_n+1)/theta/dt (plus shape derivatives, see below)
    //    d (d_n+1)
    //
    // prefactor is fac, since timefacfac/theta/dt = fac
    const double densfac = scatravarmanager_->phinp(k) * fac * densnp;
    for (unsigned vi = 0; vi < nen_; ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;
      const double v = densfac * funct_(vi);

      for (unsigned ui = 0; ui < nen_; ++ui)
        for (unsigned idim = 0; idim < nsd_; ++idim)
          emat(fvi, ui * nsd_ + idim) += v * derxy_(idim, ui);
    }
  }

  // shape derivatives associated with divergence operator
  Core::LinAlg::Matrix<nsd_, nsd_> gridvelderiv(Core::LinAlg::Initialization::zero);
  gridvelderiv.multiply_nt(evelnp_, deriv_);

  if (nsd_ == 3)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const double v0 =
          gridvelderiv(1, 0) * (deriv_(2, ui) * xjm_(1, 2) - deriv_(1, ui) * xjm_(2, 2)) +
          gridvelderiv(1, 1) * (deriv_(0, ui) * xjm_(2, 2) - deriv_(2, ui) * xjm_(0, 2)) +
          gridvelderiv(1, 2) * (deriv_(1, ui) * xjm_(0, 2) - deriv_(0, ui) * xjm_(1, 2)) +
          gridvelderiv(2, 0) * (deriv_(1, ui) * xjm_(2, 1) - deriv_(2, ui) * xjm_(1, 1)) +
          gridvelderiv(2, 1) * (deriv_(2, ui) * xjm_(0, 1) - deriv_(0, ui) * xjm_(2, 1)) +
          gridvelderiv(2, 2) * (deriv_(0, ui) * xjm_(1, 1) - deriv_(1, ui) * xjm_(0, 1));
      const double v1 =
          gridvelderiv(0, 0) * (deriv_(1, ui) * xjm_(2, 2) - deriv_(2, ui) * xjm_(1, 2)) +
          gridvelderiv(0, 1) * (deriv_(2, ui) * xjm_(0, 2) - deriv_(0, ui) * xjm_(2, 2)) +
          gridvelderiv(0, 2) * (deriv_(0, ui) * xjm_(1, 2) - deriv_(1, ui) * xjm_(0, 2)) +
          gridvelderiv(2, 0) * (deriv_(2, ui) * xjm_(1, 0) - deriv_(1, ui) * xjm_(2, 0)) +
          gridvelderiv(2, 1) * (deriv_(0, ui) * xjm_(2, 0) - deriv_(2, ui) * xjm_(0, 0)) +
          gridvelderiv(2, 2) * (deriv_(1, ui) * xjm_(0, 0) - deriv_(0, ui) * xjm_(1, 0));
      const double v2 =
          gridvelderiv(0, 0) * (deriv_(2, ui) * xjm_(1, 1) - deriv_(1, ui) * xjm_(2, 1)) +
          gridvelderiv(0, 1) * (deriv_(0, ui) * xjm_(2, 1) - deriv_(2, ui) * xjm_(0, 1)) +
          gridvelderiv(0, 2) * (deriv_(1, ui) * xjm_(0, 1) - deriv_(0, ui) * xjm_(1, 1)) +
          gridvelderiv(1, 0) * (deriv_(1, ui) * xjm_(2, 0) - deriv_(2, ui) * xjm_(1, 0)) +
          gridvelderiv(1, 1) * (deriv_(2, ui) * xjm_(0, 0) - deriv_(0, ui) * xjm_(2, 0)) +
          gridvelderiv(1, 2) * (deriv_(0, ui) * xjm_(1, 0) - deriv_(1, ui) * xjm_(0, 0));

      for (unsigned vi = 0; vi < nen_; ++vi)
      {
        const int fvi = vi * numdofpernode_ + k;
        const double v = scatravarmanager_->phinp(k) * timefacfac / J * funct_(vi);

        emat(fvi, ui * 3) += v * v0;
        emat(fvi, ui * 3 + 1) += v * v1;
        emat(fvi, ui * 3 + 2) += v * v2;
      }
    }
  }

  else if (nsd_ == 2)
  {
    for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;
      const double v = scatravarmanager_->phinp(k) * timefacfac / J * funct_(vi);

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        emat(fvi, ui * 2) +=
            v * (-gridvelderiv(1, 0) * deriv_(1, ui) + gridvelderiv(1, 1) * deriv_(0, ui));
        emat(fvi, ui * 2 + 1) +=
            v * (gridvelderiv(0, 0) * deriv_(1, ui) - gridvelderiv(0, 1) * deriv_(0, ui));
      }
    }
  }

  else
    FOUR_C_THROW("Shape derivatives not implemented for 1D problems!");
}


/*-------------------------------------------------------------------- *
 |  "shapederivatives" convective term (OD mesh)           vuong 08/14 |
 |  put it into its own function                      kremheller 07/17 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::apply_shape_derivs_conv(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const double rhsfac, const double densnp,
    const double J, const Core::LinAlg::Matrix<nsd_, 1>& gradphi,
    const Core::LinAlg::Matrix<nsd_, 1>& convelint)
{
  //----------------------------------------------------------------
  // standard Galerkin terms  -- "shapederivatives" convective term
  //----------------------------------------------------------------

  if (nsd_ == 3)
  {
    const double xjm_0_0 = xjm_(0, 0);
    const double xjm_0_1 = xjm_(0, 1);
    const double xjm_0_2 = xjm_(0, 2);
    const double xjm_1_0 = xjm_(1, 0);
    const double xjm_1_1 = xjm_(1, 1);
    const double xjm_1_2 = xjm_(1, 2);
    const double xjm_2_0 = xjm_(2, 0);
    const double xjm_2_1 = xjm_(2, 1);
    const double xjm_2_2 = xjm_(2, 2);

    {
      // gradient of scalar w.r.t. reference coordinates
      static Core::LinAlg::Matrix<nsd_, 1> refgradphi;
      refgradphi.multiply(xjm_, gradphi);

      const double refgradphi_0 = refgradphi(0);
      const double refgradphi_1 = refgradphi(1);
      const double refgradphi_2 = refgradphi(2);

      const double convelint_0 = convelint(0);
      const double convelint_1 = convelint(1);
      const double convelint_2 = convelint(2);

      const double vrhs = rhsfac * densnp / J;

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const double v00 =
            +convelint_1 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_2 - deriv_(1, ui) * xjm_2_2) +
                               refgradphi_1 * (deriv_(0, ui) * xjm_2_2 - deriv_(2, ui) * xjm_0_2) +
                               refgradphi_2 * (deriv_(1, ui) * xjm_0_2 - deriv_(0, ui) * xjm_1_2)) +
            convelint_2 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_1 - deriv_(2, ui) * xjm_1_1) +
                              refgradphi_1 * (deriv_(2, ui) * xjm_0_1 - deriv_(0, ui) * xjm_2_1) +
                              refgradphi_2 * (deriv_(0, ui) * xjm_1_1 - deriv_(1, ui) * xjm_0_1));
        const double v01 =
            +convelint_0 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_2 - deriv_(2, ui) * xjm_1_2) +
                               refgradphi_1 * (deriv_(2, ui) * xjm_0_2 - deriv_(0, ui) * xjm_2_2) +
                               refgradphi_2 * (deriv_(0, ui) * xjm_1_2 - deriv_(1, ui) * xjm_0_2)) +
            convelint_2 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_0 - deriv_(1, ui) * xjm_2_0) +
                              refgradphi_1 * (deriv_(0, ui) * xjm_2_0 - deriv_(2, ui) * xjm_0_0) +
                              refgradphi_2 * (deriv_(1, ui) * xjm_0_0 - deriv_(0, ui) * xjm_1_0));
        const double v02 =
            +convelint_0 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_1 - deriv_(1, ui) * xjm_2_1) +
                               refgradphi_1 * (deriv_(0, ui) * xjm_2_1 - deriv_(2, ui) * xjm_0_1) +
                               refgradphi_2 * (deriv_(1, ui) * xjm_0_1 - deriv_(0, ui) * xjm_1_1)) +
            convelint_1 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_0 - deriv_(2, ui) * xjm_1_0) +
                              refgradphi_1 * (deriv_(2, ui) * xjm_0_0 - deriv_(0, ui) * xjm_2_0) +
                              refgradphi_2 * (deriv_(0, ui) * xjm_1_0 - deriv_(1, ui) * xjm_0_0));

        for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;
          const double v = vrhs * funct_(vi);

          emat(fvi, ui * 3 + 0) += v * v00;
          emat(fvi, ui * 3 + 1) += v * v01;
          emat(fvi, ui * 3 + 2) += v * v02;
        }
      }
    }
  }
  else if (nsd_ == 2)
  {
    {
      // gradient of scalar w.r.t. reference coordinates
      static Core::LinAlg::Matrix<nsd_, 1> refgradphi;
      refgradphi.multiply(xjm_, gradphi);

      const double refgradphi_0 = refgradphi(0);
      const double refgradphi_1 = refgradphi(1);

      const double convelint_0 = convelint(0);
      const double convelint_1 = convelint(1);

      const double vrhs = rhsfac * densnp / J;

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const double v00 =
            +convelint_1 * (-refgradphi_0 * deriv_(1, ui) + refgradphi_1 * deriv_(0, ui));
        const double v01 =
            +convelint_0 * (refgradphi_0 * deriv_(1, ui) - refgradphi_1 * deriv_(0, ui));

        for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
        {
          const int fvi = vi * numdofpernode_ + k;
          const double v = vrhs * funct_(vi);

          emat(fvi, ui * 2 + 0) += v * v00;
          emat(fvi, ui * 2 + 1) += v * v01;
        }
      }
    }
  }
  else
    FOUR_C_THROW("shapederivatives not implemented for 1D!");
}


/*-------------------------------------------------------------------- *
 |  standard Galerkin diffusive term (OD mesh)   vuong 08/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_diff_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double diffcoeff, const double fac, const double rhsfac, const double J,
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi, const Core::LinAlg::Matrix<nsd_, 1>& convelint,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  const double vrhs = -rhsfac / J * diffcoeff;

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    double laplawf(0.0);
    get_laplacian_weak_form_rhs(laplawf, gradphi, vi);
    const double val = vrhs * laplawf;

    const int fvi = vi * numdofpernode_ + k;

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      const int fui = ui * ndofpernodemesh;

      for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
        emat(fvi, fui + udim) += val * dJ_dmesh(fui + udim);
    }
  }

  //----------------------------------------------------------------
  // standard Galerkin terms  -- "shapederivatives" diffusive term
  //----------------------------------------------------------------

  if (nsd_ == 3)
  {
    const double xjm_0_0 = xjm_(0, 0);
    const double xjm_0_1 = xjm_(0, 1);
    const double xjm_0_2 = xjm_(0, 2);
    const double xjm_1_0 = xjm_(1, 0);
    const double xjm_1_1 = xjm_(1, 1);
    const double xjm_1_2 = xjm_(1, 2);
    const double xjm_2_0 = xjm_(2, 0);
    const double xjm_2_1 = xjm_(2, 1);
    const double xjm_2_2 = xjm_(2, 2);
    {
      const double v = diffcoeff * rhsfac / J;

      const double gradphi_0 = gradphi(0);
      const double gradphi_1 = gradphi(1);
      const double gradphi_2 = gradphi(2);

      for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
      {
        const double deriv_vi_0 = deriv_(0, vi);
        const double deriv_vi_1 = deriv_(1, vi);
        const double deriv_vi_2 = deriv_(2, vi);

        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
        {
          const double v00 =
              +gradphi_1 * (deriv_vi_0 * (deriv_(2, ui) * xjm_1_2 - deriv_(1, ui) * xjm_2_2) +
                               deriv_vi_1 * (deriv_(0, ui) * xjm_2_2 - deriv_(2, ui) * xjm_0_2) +
                               deriv_vi_2 * (deriv_(1, ui) * xjm_0_2 - deriv_(0, ui) * xjm_1_2)) +
              gradphi_2 * (deriv_vi_0 * (deriv_(1, ui) * xjm_2_1 - deriv_(2, ui) * xjm_1_1) +
                              deriv_vi_1 * (deriv_(2, ui) * xjm_0_1 - deriv_(0, ui) * xjm_2_1) +
                              deriv_vi_2 * (deriv_(0, ui) * xjm_1_1 - deriv_(1, ui) * xjm_0_1));
          const double v01 =
              +gradphi_0 * (deriv_vi_0 * (deriv_(1, ui) * xjm_2_2 - deriv_(2, ui) * xjm_1_2) +
                               deriv_vi_1 * (deriv_(2, ui) * xjm_0_2 - deriv_(0, ui) * xjm_2_2) +
                               deriv_vi_2 * (deriv_(0, ui) * xjm_1_2 - deriv_(1, ui) * xjm_0_2)) +
              gradphi_2 * (deriv_vi_0 * (deriv_(2, ui) * xjm_1_0 - deriv_(1, ui) * xjm_2_0) +
                              deriv_vi_1 * (deriv_(0, ui) * xjm_2_0 - deriv_(2, ui) * xjm_0_0) +
                              deriv_vi_2 * (deriv_(1, ui) * xjm_0_0 - deriv_(0, ui) * xjm_1_0));
          const double v02 =
              +gradphi_0 * (deriv_vi_0 * (deriv_(2, ui) * xjm_1_1 - deriv_(1, ui) * xjm_2_1) +
                               deriv_vi_1 * (deriv_(0, ui) * xjm_2_1 - deriv_(2, ui) * xjm_0_1) +
                               deriv_vi_2 * (deriv_(1, ui) * xjm_0_1 - deriv_(0, ui) * xjm_1_1)) +
              gradphi_1 * (deriv_vi_0 * (deriv_(1, ui) * xjm_2_0 - deriv_(2, ui) * xjm_1_0) +
                              deriv_vi_1 * (deriv_(2, ui) * xjm_0_0 - deriv_(0, ui) * xjm_2_0) +
                              deriv_vi_2 * (deriv_(0, ui) * xjm_1_0 - deriv_(1, ui) * xjm_0_0));

          emat(fvi, ui * ndofpernodemesh + 0) += v * v00;
          emat(fvi, ui * ndofpernodemesh + 1) += v * v01;
          emat(fvi, ui * ndofpernodemesh + 2) += v * v02;
        }
      }
    }

    const double v = diffcoeff * rhsfac / J;

    // gradient of scalar w.r.t. reference coordinates
    static Core::LinAlg::Matrix<nsd_, 1> refgradphi;
    refgradphi.multiply(xjm_, gradphi);

    const double refgradphi_0 = refgradphi(0);
    const double refgradphi_1 = refgradphi(1);
    const double refgradphi_2 = refgradphi(2);

    for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    {
      const double derxy_vi_0 = derxy_(0, vi);
      const double derxy_vi_1 = derxy_(1, vi);
      const double derxy_vi_2 = derxy_(2, vi);

      const int fvi = vi * numdofpernode_ + k;

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const double v00 =
            +derxy_vi_1 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_2 - deriv_(1, ui) * xjm_2_2) +
                              refgradphi_1 * (deriv_(0, ui) * xjm_2_2 - deriv_(2, ui) * xjm_0_2) +
                              refgradphi_2 * (deriv_(1, ui) * xjm_0_2 - deriv_(0, ui) * xjm_1_2)) +
            derxy_vi_2 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_1 - deriv_(2, ui) * xjm_1_1) +
                             refgradphi_1 * (deriv_(2, ui) * xjm_0_1 - deriv_(0, ui) * xjm_2_1) +
                             refgradphi_2 * (deriv_(0, ui) * xjm_1_1 - deriv_(1, ui) * xjm_0_1));
        const double v01 =
            +derxy_vi_0 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_2 - deriv_(2, ui) * xjm_1_2) +
                              refgradphi_1 * (deriv_(2, ui) * xjm_0_2 - deriv_(0, ui) * xjm_2_2) +
                              refgradphi_2 * (deriv_(0, ui) * xjm_1_2 - deriv_(1, ui) * xjm_0_2)) +
            derxy_vi_2 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_0 - deriv_(1, ui) * xjm_2_0) +
                             refgradphi_1 * (deriv_(0, ui) * xjm_2_0 - deriv_(2, ui) * xjm_0_0) +
                             refgradphi_2 * (deriv_(1, ui) * xjm_0_0 - deriv_(0, ui) * xjm_1_0));
        const double v02 =
            +derxy_vi_0 * (refgradphi_0 * (deriv_(2, ui) * xjm_1_1 - deriv_(1, ui) * xjm_2_1) +
                              refgradphi_1 * (deriv_(0, ui) * xjm_2_1 - deriv_(2, ui) * xjm_0_1) +
                              refgradphi_2 * (deriv_(1, ui) * xjm_0_1 - deriv_(0, ui) * xjm_1_1)) +
            derxy_vi_1 * (refgradphi_0 * (deriv_(1, ui) * xjm_2_0 - deriv_(2, ui) * xjm_1_0) +
                             refgradphi_1 * (deriv_(2, ui) * xjm_0_0 - deriv_(0, ui) * xjm_2_0) +
                             refgradphi_2 * (deriv_(0, ui) * xjm_1_0 - deriv_(1, ui) * xjm_0_0));

        emat(fvi, ui * ndofpernodemesh + 0) += v * v00;
        emat(fvi, ui * ndofpernodemesh + 1) += v * v01;
        emat(fvi, ui * ndofpernodemesh + 2) += v * v02;
      }
    }
  }
  else if (nsd_ == 2)
  {
    {
      const double v = diffcoeff * rhsfac / J;

      const double gradphi_0 = gradphi(0);
      const double gradphi_1 = gradphi(1);

      for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
      {
        const double deriv_vi_0 = deriv_(0, vi);
        const double deriv_vi_1 = deriv_(1, vi);

        const int fvi = vi * numdofpernode_ + k;

        for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
        {
          const double v00 =
              +gradphi_1 * (-deriv_vi_0 * deriv_(1, ui) + deriv_vi_1 * deriv_(0, ui));
          const double v01 = +gradphi_0 * (deriv_vi_0 * deriv_(1, ui) - deriv_vi_1 * deriv_(0, ui));

          emat(fvi, ui * ndofpernodemesh + 0) += v * v00;
          emat(fvi, ui * ndofpernodemesh + 1) += v * v01;
        }
      }
    }

    const double v = diffcoeff * rhsfac / J;

    // gradient of scalar w.r.t. reference coordinates
    static Core::LinAlg::Matrix<nsd_, 1> refgradphi;
    refgradphi.multiply(xjm_, gradphi);

    const double refgradphi_0 = refgradphi(0);
    const double refgradphi_1 = refgradphi(1);

    for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    {
      const double derxy_vi_0 = derxy_(0, vi);
      const double derxy_vi_1 = derxy_(1, vi);

      const int fvi = vi * numdofpernode_ + k;

      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const double v00 =
            +derxy_vi_1 * (-refgradphi_0 * deriv_(1, ui) + refgradphi_1 * deriv_(0, ui));
        const double v01 =
            +derxy_vi_0 * (refgradphi_0 * deriv_(1, ui) - refgradphi_1 * deriv_(0, ui));

        emat(fvi, ui * ndofpernodemesh + 0) += v * v00;
        emat(fvi, ui * ndofpernodemesh + 1) += v * v01;
      }
    }
  }
  else
    FOUR_C_THROW("shapederivatives not implemented for 1D!");
}

/*-------------------------------------------------------------------- *
 |  standard Galerkin reactive term (OD mesh)              vuong 08/14 |
 *---------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_react_od_mesh(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double rea_phi, const double J,
    const Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  if (reamanager_->active())
  {
    // standard Galerkin term
    double vrhs = rhsfac * rea_phi / J;

    for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    {
      const int fvi = vi * numdofpernode_ + k;

      const double val = vrhs * funct_(vi);
      for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
      {
        const int fui = ui * ndofpernodemesh;

        for (int udim = 0; udim < static_cast<int>(nsd_); ++udim)
          emat(fvi, fui + udim) += val * dJ_dmesh(fui + udim);
      }
    }
  }
}

/*------------------------------------------------------------------- *
 |  calculation of linearized mass (OD fluid)        kremheller 07/17 |
 *--------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_lin_mass_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double fac, const double densam, const double densnp,
    const double phinp, const double hist)
{
}

/*------------------------------------------------------------------------------------------ *
 |  standard Galerkin transient, old part of rhs and source term (OD fluid) kremheller 07/17 |
 *-------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_hist_and_source_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh, const double fac,
    const double rhsint, const double densnp)
{
}

/*------------------------------------------------------------------ *
 |  standard Galerkin reactive term (OD fluid)  kremheller 07/17     |
 *----------------------------------------------------------------   */
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_react_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat, const int k, const int ndofpernodemesh,
    const double rhsfac, const double rea_phi)
{
}

/*------------------------------------------------------------------ *
 |  standard Galerkin diffusive term (OD fluid)  kremheller 07/17    |
 *----------------------------------------------------------------   */
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_diff_od_fluid(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element current to be filled
    const int k,                            //!< index of current scalar
    const int ndofpernodemesh,              //!< number of dofs per node of ale element
    const double rhsfac,  //!< time-integration factor for rhs times domain-integration factor
    const Core::LinAlg::Matrix<nsd_, 1>& gradphi  //!< scalar gradient at Gauss point
)
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype, int probdim>
void Discret::Elements::ScaTraEleCalc<distype, probdim>::calc_djd_mesh(
    Core::LinAlg::Matrix<1, nsd_ * nen_>& dJ_dmesh)
{
  const double J = xjm_.determinant();
  for (unsigned node = 0; node < nen_; node++)
    for (int dim = 0; dim < static_cast<int>(nsd_); dim++)
      dJ_dmesh(dim + node * nsd_) = J * derxy_(dim, node);
}

FOUR_C_NAMESPACE_CLOSE

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
// template classes

#include "4C_scatra_ele_calc.inst.hpp"

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_sti_electrode.hpp"

#include "4C_mat_fourier.hpp"
#include "4C_mat_soret.hpp"
#include "4C_scatra_ele_calc_elch_electrode.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_sti_thermo.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | singleton access method                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcSTIElectrode<distype>*
Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcSTIElectrode<distype>>(
            new ScaTraEleCalcSTIElectrode<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*--------------------------------------------------------------------------*
 | calculate element matrix and element right-hand side vector   fang 11/15 |
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::sysmat(
    Core::Elements::Element* ele,               ///< current element
    Core::LinAlg::SerialDenseMatrix& emat,      ///< element matrix
    Core::LinAlg::SerialDenseVector& erhs,      ///< element right-hand side vector
    Core::LinAlg::SerialDenseVector& subgrdiff  ///< subgrid diffusivity scaling vector
)
{
  // density at time t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(my::numscal_, 0.);

  // density at time t_(n+alpha_M)
  std::vector<double> densam(my::numscal_, 0.);

  // dummy variable
  std::vector<double> dummyvec(my::numscal_, 0.);
  double dummy(0.);

  // integration points and weights
  const Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(
      ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions, their derivatives, and domain integration factor at current
    // integration point
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // evaluate overall integration factors
    double timefacfac = my::scatraparatimint_->time_fac() * fac;
    double rhsfac = my::scatraparatimint_->time_fac_rhs() * fac;

    // evaluate internal variables at current integration point
    set_internal_variables_for_mat_and_rhs();

    // evaluate material parameters at current integration point
    get_material_params(ele, dummyvec, densnp, densam, dummy, iquad);

    // matrix and vector contributions arising from mass term
    if (not my::scatraparatimint_->is_stationary())
    {
      my::calc_mat_mass(emat, 0, fac, densam[0]);
      my::calc_rhs_lin_mass(erhs, 0, rhsfac, fac, densam[0], densnp[0]);
    }

    // vector contributions arising from history value
    // need to adapt history value to time integration scheme first
    double rhsint(0.0);
    my::compute_rhs_int(rhsint, densam[0], densnp[0], my::scatravarmanager_->hist(0));
    my::calc_rhs_hist_and_source(erhs, 0, fac, rhsint);

    // matrix and vector contributions arising from diffusion term
    my::calc_mat_diff(emat, 0, timefacfac);
    my::calc_rhs_diff(erhs, 0, rhsfac);


    // matrix and vector contributions arising from conservative part of convective term (deforming
    // meshes)
    if (my::scatrapara_->is_conservative())
    {
      double vdiv(0.0);
      my::get_divergence(vdiv, my::evelnp_);
      my::calc_mat_conv_add_cons(emat, 0, timefacfac, vdiv, densnp[0]);

      double vrhs = rhsfac * my::scatravarmanager_->phinp(0) * vdiv * densnp[0];
      for (unsigned vi = 0; vi < nen_; ++vi) erhs[vi * my::numdofpernode_] -= vrhs * my::funct_(vi);
    }

    // matrix and vector contributions arising from source terms
    if (ele->material()->material_type() == Core::Materials::m_soret)
      mystielch::calc_mat_and_rhs_source(emat, erhs, timefacfac, rhsfac);
    else if (ele->material()->material_type() == Core::Materials::m_thermo_fourier)
      calc_mat_and_rhs_joule(emat, erhs, timefacfac, rhsfac);
  }  // loop over integration points
}


/*------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from Joule's heat   fang 11/15 |
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_and_rhs_joule(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // square of gradient of electric potential
  const double gradpot2 = var_manager()->grad_pot().dot(var_manager()->grad_pot());

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    // linearizations of Joule's heat term in thermo residuals w.r.t. thermo dofs are zero
    // contributions of Joule's heat term to thermo residuals
    erhs[vi] += rhsfac * my::funct_(vi) * gradpot2 * diffmanagerstielectrode_->get_cond();
  }
}


/*--------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from heat of mixing   fang 11/15
 |
 *--------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_and_rhs_mixing(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerstielectrode_->get_isotropic_diff(0);
  const double& F = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);

  // ionic flux density
  Core::LinAlg::Matrix<nsd_, 1> n = var_manager()->grad_conc();
  n.update(-diffcoeff * concentration * soret / temperature, gradtemp, -diffcoeff);

  // derivative of square of ionic flux density w.r.t. temperature
  const double dn2_dT =
      2. * n.dot(gradtemp) * diffcoeff * concentration * soret / pow(temperature, 2);

  // formal, symbolic derivative of square of ionic flux density w.r.t. temperature gradient
  Core::LinAlg::Matrix<nsd_, 1> dn2_dgradT = n;
  dn2_dgradT.scale(-2. * diffcoeff * concentration * soret / temperature);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times derivative of square of ionic flux density w.r.t.
      // temperature gradient
      double dn2_dgradT_ui(0.);
      my::get_laplacian_weak_form_rhs(dn2_dgradT_ui, dn2_dgradT, ui);

      // linearizations of heat of mixing term in thermo residuals w.r.t. thermo dofs
      emat(vi, ui) += timefacfac * my::funct_(vi) * F * diffmanagerstielectrode_->get_ocp_deriv() /
                      diffcoeff * (my::funct_(ui) * dn2_dT + dn2_dgradT_ui);
    }

    // contributions of heat of mixing term to thermo residuals
    erhs[vi] -= rhsfac * diffmanagerstielectrode_->get_ocp_deriv() * F * n.dot(n) / diffcoeff *
                my::funct_(vi);
  }
}


/*------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from Soret effect   fang 11/15 |
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_and_rhs_soret(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerstielectrode_->get_isotropic_diff(0);
  const double& F = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);

  // ionic flux density
  Core::LinAlg::Matrix<nsd_, 1> n = var_manager()->grad_conc();
  n.update(-diffcoeff * concentration * soret / temperature, gradtemp, -diffcoeff);

  // derivative of ionic flux density w.r.t. temperature
  Core::LinAlg::Matrix<nsd_, 1> dn_dT = gradtemp;
  dn_dT.scale(diffcoeff * concentration * soret / pow(temperature, 2));

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    // gradient of test function times ionic flux density
    double laplawfrhs_n_vi(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_n_vi, n, vi);

    // gradient of test function times derivative of ionic flux density w.r.t. temperature
    double laplawfrhs_dndT(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_dndT, dn_dT, vi);

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times gradient of test function
      double laplawf(0.);
      my::get_laplacian_weak_form(laplawf, vi, ui);

      // gradient of shape function times gradient of temperature
      double laplawfrhs_gradtemp(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp, gradtemp, ui);

      // gradient of shape function times gradient of ionic flux density
      double laplawfrhs_n_ui(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_n_ui, n, ui);

      // linearizations of Soret effect term in thermo residuals w.r.t. thermo dofs
      emat(vi, ui) += timefacfac * my::funct_(vi) * F * concentration * soret *
                          diffmanagerstielectrode_->get_ocp_deriv() *
                          (laplawfrhs_n_ui / temperature +
                              laplawfrhs_gradtemp / temperature * (-diffcoeff) * concentration *
                                  soret / temperature -
                              gradtemp.dot(n) * pow(1 / temperature, 2.0) * my::funct_(ui) +
                              gradtemp.dot(dn_dT) * my::funct_(ui) / temperature) +
                      timefacfac * F * concentration * soret *
                          diffmanagerstielectrode_->get_ocp_deriv() *
                          (-laplawf * diffcoeff * concentration * soret / temperature +
                              laplawfrhs_dndT * my::funct_(ui));
    }

    // contributions of Soret effect term to thermo residuals
    erhs[vi] -= rhsfac * concentration * diffmanagerstielectrode_->get_ocp_deriv() * F * soret *
                (my::funct_(vi) * n.dot(gradtemp) / temperature + laplawfrhs_n_vi);
  }
}


/*----------------------------------------------------------------------*
 | evaluate action for off-diagonal system matrix block      fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::evaluate_action_od(
    Core::Elements::Element* ele,                     //!< current element
    Teuchos::ParameterList& params,                   //!< parameter list
    Core::FE::Discretization& discretization,         //!< discretization
    const ScaTra::Action& action,                     //!< action parameter
    Core::Elements::LocationArray& la,                //!< location array
    Core::LinAlg::SerialDenseMatrix& elemat1_epetra,  //!< element matrix 1
    Core::LinAlg::SerialDenseMatrix& elemat2_epetra,  //!< element matrix 2
    Core::LinAlg::SerialDenseVector& elevec1_epetra,  //!< element right-hand side vector 1
    Core::LinAlg::SerialDenseVector& elevec2_epetra,  //!< element right-hand side vector 2
    Core::LinAlg::SerialDenseVector& elevec3_epetra   //!< element right-hand side vector 3
)
{
  // determine and evaluate action
  switch (action)
  {
    case ScaTra::Action::calc_scatra_mono_odblock_thermoscatra:
    {
      sysmat_od_thermo_scatra(ele, elemat1_epetra);

      break;
    }

    default:
    {
      // call base class routine
      my::evaluate_action_od(ele, params, discretization, action, la, elemat1_epetra,
          elemat2_epetra, elevec1_epetra, elevec2_epetra, elevec3_epetra);

      break;
    }
  }  // switch(action)

  return 0;
}


/*------------------------------------------------------------------------------------------------------*
 | fill element matrix with linearizations of discrete thermo residuals w.r.t. scatra dofs   fang
 11/15 |
 *------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::sysmat_od_thermo_scatra(
    Core::Elements::Element* ele,          //!< current element
    Core::LinAlg::SerialDenseMatrix& emat  //!< element matrix
)
{
  // integration points and weights
  Core::FE::IntPointsAndWeights<nsd_ele_> intpoints(ScaTra::DisTypeToOptGaussRule<distype>::rule);

  // loop over integration points
  for (int iquad = 0; iquad < intpoints.ip().nquad; ++iquad)
  {
    // evaluate shape functions, their derivatives, and domain integration factor at current
    // integration point
    const double fac = my::eval_shape_func_and_derivs_at_int_point(intpoints, iquad);

    // evaluate internal variables at current integration point
    set_internal_variables_for_mat_and_rhs();

    // evaluate material parameters at current integration point
    double dummy(0.);
    std::vector<double> dummyvec(my::numscal_, 0.);
    get_material_params(ele, dummyvec, dummyvec, dummyvec, dummy, iquad);

    // provide element matrix with linearizations of source terms in discrete thermo residuals
    // w.r.t. scatra dofs
    if (ele->material()->material_type() == Core::Materials::m_soret)
      mystielch::calc_mat_source_od(emat, my::scatraparatimint_->time_fac() * fac);
    else if (ele->material()->material_type() == Core::Materials::m_thermo_fourier)
      calc_mat_joule_od(emat, my::scatraparatimint_->time_fac() * fac);
  }
}


/*------------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of Joule's heat term in discrete thermo residuals
 w.r.t. scatra dofs   fang 11/15 |
 *------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_joule_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const Core::LinAlg::Matrix<nsd_, 1>& gradpot = var_manager()->grad_pot();
  const double gradpot2 = gradpot.dot(gradpot);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times gradient of electric potential
      double laplawfrhs_gradpot(0.0);
      my::get_laplacian_weak_form_rhs(laplawfrhs_gradpot, gradpot, ui);

      // linearizations of Joule's heat term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) -= timefacfac * my::funct_(vi) *
                          diffmanagerstielectrode_->get_conc_deriv_cond(0) * gradpot2 *
                          my::funct_(ui);

      // linearizations of Joule's heat term in thermo residuals w.r.t. electric potential dofs
      emat(vi, ui * 2 + 1) -= timefacfac * my::funct_(vi) * 2. *
                              diffmanagerstielectrode_->get_cond() * laplawfrhs_gradpot;
    }
  }
}


/*--------------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of heat of mixing term in discrete thermo residuals
 w.r.t. scatra dofs   fang 11/15 |
 *--------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_mixing_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerstielectrode_->get_isotropic_diff(0);
  const double& diffcoeffderiv = diffmanagerstielectrode_->get_conc_deriv_iso_diff_coef(0, 0);
  const Core::LinAlg::Matrix<nsd_, 1>& gradconc = var_manager()->grad_conc();
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);

  // ionic flux density
  Core::LinAlg::Matrix<nsd_, 1> n = gradconc;
  n.update(-diffcoeff * concentration * soret / temperature, gradtemp, -diffcoeff);

  // derivative of ionic flux density w.r.t. concentration
  Core::LinAlg::Matrix<nsd_, 1> dn_dc = gradconc;
  dn_dc.update(
      -diffcoeffderiv * concentration * soret / temperature - diffcoeff * soret / temperature,
      gradtemp, -diffcoeffderiv);

  // square of ionic flux density
  const double n2 = n.dot(n);

  // derivative of square of ionic flux density w.r.t. concentration
  double dn2_dc = 2. * n.dot(dn_dc);

  // derivative of square of ionic flux density w.r.t. concentration gradient
  Core::LinAlg::Matrix<nsd_, 1> dn2_dgradc = n;
  dn2_dgradc.scale(-2. * diffcoeff);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times derivative of square of ionic flux density w.r.t.
      // concentration gradient
      double dn2_dgradc_ui(0.);
      my::get_laplacian_weak_form_rhs(dn2_dgradc_ui, dn2_dgradc, ui);

      // intermediate terms
      const double term1 = diffmanagerstielectrode_->get_ocp_deriv2() * n2 * my::funct_(ui);
      const double term2 = diffmanagerstielectrode_->get_ocp_deriv() * dn2_dc * my::funct_(ui);
      const double term3 = diffmanagerstielectrode_->get_ocp_deriv() * dn2_dgradc_ui;
      const double term4 = -diffmanagerstielectrode_->get_ocp_deriv() * n2 / diffcoeff *
                           diffcoeffderiv * my::funct_(ui);

      // linearizations of heat of mixing term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) += timefacfac * my::funct_(vi) *
                          Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday() /
                          diffcoeff * (term1 + term2 + term3 + term4);

      // linearizations of heat of mixing term in thermo residuals w.r.t. electric potential dofs
      // are zero
    }
  }
}


/*------------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of Soret effect term in discrete thermo residuals
 w.r.t. scatra dofs   fang 11/15 |
 *------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::calc_mat_soret_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerstielectrode_->get_isotropic_diff(0);
  const double& diffcoeffderiv = diffmanagerstielectrode_->get_conc_deriv_iso_diff_coef(0, 0);
  const double& F = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday();
  const Core::LinAlg::Matrix<nsd_, 1>& gradconc = var_manager()->grad_conc();
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);

  // ionic flux density
  Core::LinAlg::Matrix<nsd_, 1> n = gradconc;
  n.update(-diffcoeff * concentration * soret / temperature, gradtemp, -diffcoeff);

  // derivative of ionic flux density w.r.t. concentration
  Core::LinAlg::Matrix<nsd_, 1> dn_dc = gradconc;
  dn_dc.update(
      -diffcoeffderiv * concentration * soret / temperature - diffcoeff * soret / temperature,
      gradtemp, -diffcoeffderiv);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    // gradient of test function times ionic flux density
    double laplawfrhs_n(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_n, n, vi);

    // gradient of test function times derivative of ionic flux density w.r.t. concentration
    double laplawfrhs_dndc(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_dndc, dn_dc, vi);

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times gradient of test function
      double laplawf(0.);
      my::get_laplacian_weak_form(laplawf, vi, ui);

      // gradient of shape function times temperature gradient
      double laplawfrhs_gradtemp(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp, gradtemp, ui);

      // linearizations of Soret effect term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) +=
          timefacfac * my::funct_(vi) * F * soret / temperature *
              (diffmanagerstielectrode_->get_ocp_deriv() * gradtemp.dot(n) * my::funct_(ui) +
                  concentration * diffmanagerstielectrode_->get_ocp_deriv2() * gradtemp.dot(n) *
                      my::funct_(ui) +
                  concentration * diffmanagerstielectrode_->get_ocp_deriv() * gradtemp.dot(dn_dc) *
                      my::funct_(ui) +
                  concentration * diffmanagerstielectrode_->get_ocp_deriv() * (-diffcoeff) *
                      laplawfrhs_gradtemp) +
          timefacfac * F * soret *
              (my::funct_(ui) * diffmanagerstielectrode_->get_ocp_deriv() * laplawfrhs_n +
                  my::funct_(ui) * concentration * diffmanagerstielectrode_->get_ocp_deriv2() *
                      laplawfrhs_n +
                  my::funct_(ui) * concentration * diffmanagerstielectrode_->get_ocp_deriv() *
                      laplawfrhs_dndc +
                  laplawf * concentration * diffmanagerstielectrode_->get_ocp_deriv() *
                      (-diffcoeff));

      // linearizations of Soret effect term in thermo residuals w.r.t. electric potential dofs are
      // zero
    }
  }
}


/*----------------------------------------------------------------------*
 | extract quantities for element evaluation                 fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::extract_element_and_node_values(
    Core::Elements::Element* ele,              //!< current element
    Teuchos::ParameterList& params,            //!< parameter list
    Core::FE::Discretization& discretization,  //!< discretization
    Core::Elements::LocationArray& la          //!< location array
)
{
  // call base class routine to extract thermo-related quantities
  my::extract_element_and_node_values(ele, params, discretization, la);

  // call base class routine to extract scatra-related quantities
  mystielch::extract_element_and_node_values(ele, params, discretization, la);
}


/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::get_material_params(
    const Core::Elements::Element* ele, std::vector<double>& densn, std::vector<double>& densnp,
    std::vector<double>& densam, double& visc, const int iquad)
{
  // get parameters of primary, thermal material
  std::shared_ptr<const Core::Mat::Material> material = ele->material();
  if (material->material_type() == Core::Materials::m_soret)
    mat_soret(material, densn[0], densnp[0], densam[0]);
  else if (material->material_type() == Core::Materials::m_thermo_fourier)
    mat_fourier(material, densn[0], densnp[0], densam[0]);
  else
    FOUR_C_THROW("Invalid thermal material!");

  // get parameters of secondary, scatra material
  material = ele->material(1);
  if (material->material_type() == Core::Materials::m_electrode)
  {
    utils_->mat_electrode(
        material, var_manager()->conc(), my::scatravarmanager_->phinp(0), diffmanagerstielectrode_);
    diffmanagerstielectrode_->set_ocp_and_derivs(
        ele, var_manager()->conc(), my::scatravarmanager_->phinp(0));
  }
  else
    FOUR_C_THROW("Invalid scalar transport material!");
}


/*----------------------------------------------------------------------*
 | evaluate Soret material                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::mat_soret(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< Soret material
    double& densn,                                              //!< density at time t_(n)
    double& densnp,  //!< density at time t_(n+1) or t_(n+alpha_F)
    double& densam   //!< density at time t_(n+alpha_M)
)
{
  // extract material parameters from Soret material
  const std::shared_ptr<const Mat::Soret> matsoret =
      std::static_pointer_cast<const Mat::Soret>(material);
  densn = densnp = densam = matsoret->capacity();

  std::vector<double> k = matsoret->conductivity();
  FOUR_C_ASSERT(k.size() != 1, "Conductivity value is a vector quantity, but has to be a scalar.");

  diff_manager()->set_isotropic_diff(k[0], 0);

  diff_manager()->set_soret(matsoret->soret_coefficient());
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::mat_fourier(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< Fourie material
    double& densn,                                              //!< density at time t_(n)
    double& densnp,  //!< density at time t_(n+1) or t_(n+alpha_F)
    double& densam   //!< density at time t_(n+alpha_M)
)
{
  // extract material parameters from Soret material
  const std::shared_ptr<const Mat::Fourier> matfourier =
      std::static_pointer_cast<const Mat::Fourier>(material);
  densn = densnp = densam = matfourier->capacity();

  std::vector<double> k = matfourier->conductivity();
  FOUR_C_ASSERT(k.size() != 1, "Conductivity value is a vector quantity, but has to be a scalar.");

  diff_manager()->set_isotropic_diff(k[0], 0);
}


/*------------------------------------------------------------------------------*
 | set internal variables for element evaluation                     fang 11/15 |
 *------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::set_internal_variables_for_mat_and_rhs()
{
  // set internal variables for element evaluation
  var_manager()->set_internal_variables_sti_elch(my::funct_, my::derxy_, my::ephinp_, my::ephin_,
      mystielch::econcnp_, mystielch::epotnp_, my::econvelnp_, my::ehist_);
}


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcSTIElectrode<distype>::ScaTraEleCalcSTIElectrode(
    const int numdofpernode, const int numscal, const std::string& disname)
    :  // constructors of base classes
      ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      ScaTraEleSTIElch<distype>::ScaTraEleSTIElch(numdofpernode, numscal, disname),

      // diffusion manager for electrodes
      diffmanagerstielectrode_(
          std::make_shared<ScaTraEleDiffManagerSTIElchElectrode>(my::numscal_)),

      // utility class supporting element evaluation for electrodes
      utils_(Discret::Elements::ScaTraEleUtilsElchElectrode<distype>::instance(
          numdofpernode, numscal, disname))
{
  // safety check
  if (numscal != 1 or numdofpernode != 1)
    FOUR_C_THROW("Invalid number of transported scalars or degrees of freedom per node!");

  // replace diffusion manager for standard scalar transport by thermo diffusion manager
  my::diffmanager_ = std::make_shared<ScaTraEleDiffManagerSTIThermo>(my::numscal_);

  // replace internal variable manager for standard scalar transport by internal variable manager
  // for heat transport within electrochemical substances
  my::scatravarmanager_ =
      std::make_shared<ScaTraEleInternalVariableManagerSTIElch<nsd_, nen_>>(my::numscal_);
}


// template classes
// 1D elements
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::tet10>;
// template class
// Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::pyramid5>;
// template class
// Discret::Elements::ScaTraEleCalcSTIElectrode<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_scatra_ele_calc_sti_diffcond.hpp"

#include "4C_mat_fourier.hpp"
#include "4C_mat_soret.hpp"
#include "4C_scatra_ele_calc_elch_diffcond.hpp"
#include "4C_scatra_ele_parameter_std.hpp"
#include "4C_scatra_ele_parameter_timint.hpp"
#include "4C_scatra_ele_sti_thermo.hpp"
#include "4C_scatra_ele_utils_elch_diffcond.hpp"
#include "4C_utils_singleton_owner.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 | singleton access method                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>*
Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::instance(
    const int numdofpernode, const int numscal, const std::string& disname)
{
  static auto singleton_map = Core::Utils::make_singleton_map<std::string>(
      [](const int numdofpernode, const int numscal, const std::string& disname)
      {
        return std::unique_ptr<ScaTraEleCalcSTIDiffCond<distype>>(
            new ScaTraEleCalcSTIDiffCond<distype>(numdofpernode, numscal, disname));
      });

  return singleton_map[disname].instance(
      Core::Utils::SingletonAction::create, numdofpernode, numscal, disname);
}


/*--------------------------------------------------------------------------*
 | calculate element matrix and element right-hand side vector   fang 11/15 |
 *--------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::sysmat(
    Core::Elements::Element* ele,               ///< current element
    Core::LinAlg::SerialDenseMatrix& emat,      ///< element matrix
    Core::LinAlg::SerialDenseVector& erhs,      ///< element right-hand side vector
    Core::LinAlg::SerialDenseVector& subgrdiff  ///< subgrid diffusivity scaling vector
)
{
  // density at t_(n)
  std::vector<double> densn(my::numscal_, 1.0);
  // density at t_(n+1) or t_(n+alpha_F)
  std::vector<double> densnp(my::numscal_, 1.0);
  // density at t_(n+alpha_M)
  std::vector<double> densam(my::numscal_, 1.0);

  // dummy variable
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
    get_material_params(ele, densn, densnp, densam, dummy, iquad);

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
      calc_mat_and_rhs_joule_solid(emat, erhs, timefacfac, rhsfac);
  }  // loop over integration points
}


/*------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from Joule's heat   fang 11/15 |
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_and_rhs_joule(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& invfval =
      1. / (diffmanagerdiffcond_->get_valence(0) *
               Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday());
  const double& kappa = diffmanagerdiffcond_->get_cond();
  const double& R = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
  const double& t = diffmanagerdiffcond_->get_trans_num(0);

  // current density
  Core::LinAlg::Matrix<nsd_, 1> i = var_manager()->grad_pot();
  i.update((1 - t) * invfval * 2. * R * my::scatravarmanager_->phinp(0) / concentration,
      var_manager()->grad_conc(), invfval * R * log(concentration),
      my::scatravarmanager_->grad_phi(0), -1.);
  i.scale(kappa);

  // derivative of current density w.r.t. temperature
  Core::LinAlg::Matrix<nsd_, 1> di_dT = var_manager()->grad_conc();
  di_dT.scale(kappa * (1 - t) * invfval * 2. * R / concentration);

  // formal, symbolic derivative of current density w.r.t. temperature gradient
  const double di_dgradT = kappa * invfval * R * log(concentration);

  // derivative of square of current density w.r.t. temperature gradient
  Core::LinAlg::Matrix<nsd_, 1> di2_dgradT = i;
  di2_dgradT.scale(2. * di_dgradT);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times derivative of square of current density w.r.t. temperature
      // gradient
      double di2_dgradT_gradN(0.);
      my::get_laplacian_weak_form_rhs(di2_dgradT_gradN, di2_dgradT, ui);

      // linearizations of Joule's heat term in thermo residuals w.r.t. thermo dofs
      emat(vi, ui) -= timefacfac * my::funct_(vi) / kappa *
                      (di2_dgradT_gradN + 2. * i.dot(di_dT) * my::funct_(ui));
    }

    // contributions of Joule's heat term to thermo residuals
    erhs[vi] += rhsfac * my::funct_(vi) * i.dot(i) / kappa;
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_and_rhs_joule_solid(
    Core::LinAlg::SerialDenseMatrix& emat, Core::LinAlg::SerialDenseVector& erhs,
    const double& timefacfac, const double& rhsfac)
{
  // no contributions to matrix

  // square of gradient of electric potential
  const double gradpot2 = var_manager()->grad_pot().dot(var_manager()->grad_pot());

  // linearizations of Joule's heat term in thermo residuals w.r.t. thermo dofs are zero
  // contributions of Joule's heat term to thermo residuals
  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
    erhs[vi] += rhsfac * my::funct_(vi) * gradpot2 * diffmanagerdiffcond_->get_cond();
}


/*--------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from heat of mixing   fang 11/15
 |
 *--------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_and_rhs_mixing(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerdiffcond_->get_isotropic_diff(0);
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // gradient of concentration plus scaled gradient of temperature
  Core::LinAlg::Matrix<nsd_, 1> a = var_manager()->grad_conc();
  a.update(concentration * soret / temperature, gradtemp, 1.);

  // square of abovementioned gradient
  const double a2 = a.dot(a);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // abovementioned gradient times gradient of shape function
      double laplawfrhs_a(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_a, a, ui);

      // intermediate terms
      const double term1 = 1. / concentration * a2 * my::funct_(ui);
      const double term2 =
          -2. * temperature * a.dot(gradtemp) * soret * pow(1 / temperature, 2) * my::funct_(ui);
      const double term3 = 2. * temperature * laplawfrhs_a * soret / temperature;

      // linearizations of heat of mixing term in thermo residuals w.r.t. thermo dofs
      emat(vi, ui) -= timefacfac * my::funct_(vi) * pow(diffcoeff, 2) * 2. * gasconstant *
                      (term1 + term2 + term3);
    }

    // contributions of heat of mixing term to thermo residuals
    erhs[vi] += rhsfac * my::funct_(vi) * pow(diffcoeff, 2) * gasconstant * 2. * temperature /
                concentration * a2;
  }
}


/*------------------------------------------------------------------------------------------------*
 | element matrix and right-hand side vector contributions arising from Soret effect   fang 11/15 |
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_and_rhs_soret(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    Core::LinAlg::SerialDenseVector& erhs,  //!< element right-hand side vector
    const double& timefacfac,  //!< domain integration factor times time integration factor
    const double& rhsfac       //!< domain integration factor times time integration factor for
                               //!< right-hand side vector
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerdiffcond_->get_isotropic_diff(0);
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& R = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);

  // gradient of concentration plus scaled gradient of temperature
  Core::LinAlg::Matrix<nsd_, 1> a = var_manager()->grad_conc();
  a.update(concentration * soret / temperature, gradtemp, 1.);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    // abovementioned gradient times gradient of test function
    double laplawfrhs_a_vi(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_a_vi, a, vi);

    // temperature gradient times gradient of test function
    double laplawfrhs_gradtemp_vi(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp_vi, gradtemp, vi);

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // abovementioned gradient times gradient of shape function
      double laplawfrhs_a_ui(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_a_ui, a, ui);

      // temperature gradient times gradient of shape function
      double laplawfrhs_gradtemp_ui(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp_ui, gradtemp, ui);

      // gradient of test function times gradient of shape function
      double laplawf(0.);
      my::get_laplacian_weak_form(laplawf, ui, vi);

      // intermediate terms
      const double term1 = -gradtemp.dot(gradtemp) * soret / pow(temperature, 2) * my::funct_(ui);
      const double term2 = laplawfrhs_a_ui / concentration;
      const double term3 = laplawfrhs_gradtemp_ui * soret / temperature;
      const double term4 = my::funct_(ui) * laplawfrhs_a_vi / concentration;
      const double term5 = -soret / temperature * laplawfrhs_gradtemp_vi * my::funct_(ui);
      const double term6 = soret * laplawf;

      // linearizations of Soret effect term in thermo residuals w.r.t. thermo dofs
      emat(vi, ui) += timefacfac * diffcoeff * concentration * 2. * R * soret *
                      ((term1 + term2 + term3) * my::funct_(vi) + term4 + term5 + term6);
    }

    // contributions of Soret effect term to thermo residuals
    erhs[vi] -= rhsfac * diffcoeff * 2. * R * soret *
                (a.dot(gradtemp) * my::funct_(vi) + temperature * laplawfrhs_a_vi);
  }
}


/*----------------------------------------------------------------------*
 | evaluate action for off-diagonal system matrix block      fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
int Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::evaluate_action_od(
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
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::sysmat_od_thermo_scatra(
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
    std::vector<double> dummy(my::numscal_, 0.);
    double dummy2(0.);
    get_material_params(ele, dummy, dummy, dummy, dummy2, iquad);

    // provide element matrix with linearizations of source terms in discrete thermo residuals
    // w.r.t. scatra dofs
    if (ele->material()->material_type() == Core::Materials::m_soret)
      mystielch::calc_mat_source_od(emat, my::scatraparatimint_->time_fac() * fac);
    else if (ele->material()->material_type() == Core::Materials::m_thermo_fourier)
      calc_mat_joule_solid_od(emat, my::scatraparatimint_->time_fac() * fac);
  }
}


/*------------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of Joule's heat term in discrete thermo residuals
 w.r.t. scatra dofs   fang 11/15 |
 *------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_joule_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const Core::LinAlg::Matrix<nsd_, 1>& gradconc = var_manager()->grad_conc();
  const Core::LinAlg::Matrix<nsd_, 1>& gradpot = var_manager()->grad_pot();
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double invfval =
      1. / (diffmanagerdiffcond_->get_valence(0) *
               Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday());
  const double& kappa = diffmanagerdiffcond_->get_cond();
  const double& kappaderiv = diffmanagerdiffcond_->get_conc_deriv_cond(0);
  const double& R = Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();
  const double& t = diffmanagerdiffcond_->get_trans_num(0);
  const double& temperature = my::scatravarmanager_->phinp(0);

  // current density
  Core::LinAlg::Matrix<nsd_, 1> i = gradpot;
  i.update((1 - t) * invfval * 2. * R * temperature / concentration, gradconc,
      invfval * R * log(concentration), gradtemp, -1.);
  i.scale(kappa);

  // derivative of current density w.r.t. concentration
  Core::LinAlg::Matrix<nsd_, 1> di_dc = gradpot;
  di_dc.update(kappaderiv * (1 - t) * invfval * 2. * R * temperature / concentration -
                   kappa * diffmanagerdiffcond_->get_deriv_trans_num(0, 0) * invfval * 2. * R *
                       temperature / concentration -
                   kappa * (1 - t) * invfval * 2. * R * temperature / pow(concentration, 2),
      gradconc, kappaderiv * invfval * R * log(concentration) + kappa * invfval * R / concentration,
      gradtemp, -kappaderiv);

  // formal, symbolic derivative of current density w.r.t. concentration gradient
  const double di_dgradc = kappa * (1 - t) * invfval * 2. * R * temperature / concentration;

  // square of current density
  const double i2 = i.dot(i);

  // derivative of square of current density w.r.t. concentration
  const double di2_dc = 2. * i.dot(di_dc);

  // derivative of square of current density w.r.t. concentration gradient
  Core::LinAlg::Matrix<nsd_, 1> di2_dgradc = i;
  di2_dgradc.scale(2. * di_dgradc);

  // derivative of square of current density w.r.t. gradient of electric potential
  Core::LinAlg::Matrix<nsd_, 1> di2_dgradpot = i;
  di2_dgradpot.scale(-2. * kappa);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times derivative of square of current density w.r.t.
      // concentration gradient
      double di2_dgradc_gradN(0.);
      my::get_laplacian_weak_form_rhs(di2_dgradc_gradN, di2_dgradc, ui);

      // gradient of shape function times derivative of square of current density w.r.t. gradient of
      // electric potential
      double di2_dgradpot_gradN(0.0);
      my::get_laplacian_weak_form_rhs(di2_dgradpot_gradN, di2_dgradpot, ui);

      // intermediate terms
      const double term1 = my::funct_(ui) * di2_dc;
      const double term2 = di2_dgradc_gradN;
      const double term3 = -my::funct_(ui) * kappaderiv * i2 / kappa;

      // linearizations of Joule's heat term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) -= timefacfac * my::funct_(vi) * (term1 + term2 + term3) / kappa;

      // linearizations of Joule's heat term in thermo residuals w.r.t. electric potential dofs
      emat(vi, ui * 2 + 1) -= timefacfac * my::funct_(vi) * di2_dgradpot_gradN / kappa;
    }
  }
}

/*------------------------------------------------------------------------------------------------*
 *------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_joule_solid_od(
    Core::LinAlg::SerialDenseMatrix& emat, const double& timefacfac)
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

      // linearizations of Joule's heat term in thermo residuals w.r.t. concentration dofs (in case
      // conductivity is a function of the concentration)
      emat(vi, ui * 2) -= timefacfac * my::funct_(vi) *
                          diffmanagerdiffcond_->get_conc_deriv_cond(0) * gradpot2 * my::funct_(ui);

      // linearizations of Joule's heat term in thermo residuals w.r.t. electric potential dofs
      emat(vi, ui * 2 + 1) -=
          timefacfac * my::funct_(vi) * 2.0 * diffmanagerdiffcond_->get_cond() * laplawfrhs_gradpot;
    }
  }
}


/*--------------------------------------------------------------------------------------------------------------------------------*
 | provide element matrix with linearizations of heat of mixing term in discrete thermo residuals
 w.r.t. scatra dofs   fang 11/15 |
 *--------------------------------------------------------------------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_mixing_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerdiffcond_->get_isotropic_diff(0);
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // gradient of concentration plus scaled gradient of temperature
  Core::LinAlg::Matrix<nsd_, 1> a = var_manager()->grad_conc();
  a.update(concentration * soret / temperature, gradtemp, 1.);

  // square of abovementioned gradient
  const double a2 = a.dot(a);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // abovementioned gradient times gradient of shape function
      double laplawfrhs_a(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_a, a, ui);

      // intermediate terms
      const double term1 = 2. * diffcoeff / concentration *
                           diffmanagerdiffcond_->get_conc_deriv_iso_diff_coef(0, 0) * a2 *
                           my::funct_(ui);
      const double term2 = -pow(diffcoeff, 2) / pow(concentration, 2) * a2 * my::funct_(ui);
      const double term3 = 2. * pow(diffcoeff, 2) / concentration * a.dot(gradtemp) * soret /
                           temperature * my::funct_(ui);
      const double term4 = 2. * pow(diffcoeff, 2) / concentration * laplawfrhs_a;

      // linearizations of heat of mixing term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) += -timefacfac * my::funct_(vi) * gasconstant * temperature * 2. *
                          (term1 + term2 + term3 + term4);

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
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::calc_mat_soret_od(
    Core::LinAlg::SerialDenseMatrix& emat,  //!< element matrix
    const double& timefacfac  //!< domain integration factor times time integration factor
)
{
  // extract variables and parameters
  const double& concentration = var_manager()->conc();
  const double& diffcoeff = diffmanagerdiffcond_->get_isotropic_diff(0);
  const double& diffcoeffderiv = diffmanagerdiffcond_->get_conc_deriv_iso_diff_coef(0, 0);
  const Core::LinAlg::Matrix<nsd_, 1>& gradtemp = my::scatravarmanager_->grad_phi(0);
  const double& soret = diff_manager()->get_soret();
  const double& temperature = my::scatravarmanager_->phinp(0);
  const double gasconstant =
      Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant();

  // square of temperature gradient
  const double gradtemp2 = gradtemp.dot(gradtemp);

  // gradient of concentration plus scaled gradient of temperature
  Core::LinAlg::Matrix<nsd_, 1> a = var_manager()->grad_conc();
  a.update(concentration * soret / temperature, gradtemp, 1.);

  // abovementioned gradient times temperature gradient
  const double gradtemp_a = gradtemp.dot(a);

  for (int vi = 0; vi < static_cast<int>(nen_); ++vi)
  {
    // gradient of test function times abovementioned gradient
    double laplawfrhs_a(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_a, a, vi);

    // gradient of test function times temperature gradient
    double laplawfrhs_gradtemp_vi(0.);
    my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp_vi, gradtemp, vi);

    for (int ui = 0; ui < static_cast<int>(nen_); ++ui)
    {
      // gradient of shape function times temperature gradient
      double laplawfrhs_gradtemp_ui(0.);
      my::get_laplacian_weak_form_rhs(laplawfrhs_gradtemp_ui, gradtemp, ui);

      // gradient of test function times gradient of shape function
      double laplawf(0.);
      my::get_laplacian_weak_form(laplawf, vi, ui);

      // linearizations of Soret effect term in thermo residuals w.r.t. concentration dofs
      emat(vi, ui * 2) +=
          timefacfac * soret * 2. * gasconstant *
          (my::funct_(ui) *
                  (diffcoeffderiv * (gradtemp_a * my::funct_(vi) + temperature * laplawfrhs_a) +
                      diffcoeff * soret *
                          (gradtemp2 * my::funct_(vi) / temperature + laplawfrhs_gradtemp_vi)) +
              diffcoeff * (laplawfrhs_gradtemp_ui * my::funct_(vi) + temperature * laplawf));

      // linearizations of Soret effect term in thermo residuals w.r.t. electric potential dofs are
      // zero
    }
  }
}


/*----------------------------------------------------------------------*
 | extract quantities for element evaluation                 fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::extract_element_and_node_values(
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
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::get_material_params(
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
  if (material->material_type() == Core::Materials::m_elchmat)
  {
    // pre calculate RT and F^2/(RT)
    const double rt =
        Discret::Elements::ScaTraEleParameterElch::instance("scatra")->gas_constant() *
        var_manager()->phinp(0);
    const double ffrt =
        std::pow(Discret::Elements::ScaTraEleParameterElch::instance("scatra")->faraday(), 2) / rt;

    std::vector<double> concentrations(1, var_manager()->conc());
    Inpar::ElCh::DiffCondMat dummy(Inpar::ElCh::diffcondmat_undefined);
    utils_->mat_elch_mat(material, concentrations, var_manager()->phinp(0),
        Inpar::ElCh::equpot_undefined, ffrt, diffmanagerdiffcond_, dummy);
  }
  else
    FOUR_C_THROW("Invalid scalar transport material!");
}

/*----------------------------------------------------------------------*
 | evaluate Soret material                                   fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::mat_soret(
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
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::mat_fourier(
    const std::shared_ptr<const Core::Mat::Material> material,  //!< Fourier material
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
void Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::set_internal_variables_for_mat_and_rhs()
{
  // set internal variables for element evaluation
  var_manager()->set_internal_variables_sti_elch(my::funct_, my::derxy_, my::ephinp_, my::ephin_,
      mystielch::econcnp_, mystielch::epotnp_, my::econvelnp_, my::ehist_);
}  // Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::set_internal_variables_for_mat_and_rhs


/*----------------------------------------------------------------------*
 | private constructor for singletons                        fang 11/15 |
 *----------------------------------------------------------------------*/
template <Core::FE::CellType distype>
Discret::Elements::ScaTraEleCalcSTIDiffCond<distype>::ScaTraEleCalcSTIDiffCond(
    const int numdofpernode, const int numscal, const std::string& disname)
    :  // constructors of base classes
      ScaTraEleCalc<distype>::ScaTraEleCalc(numdofpernode, numscal, disname),
      ScaTraEleSTIElch<distype>::ScaTraEleSTIElch(numdofpernode, numscal, disname),

      // diffusion manager for diffusion-conduction formulation
      diffmanagerdiffcond_(std::make_shared<ScaTraEleDiffManagerElchDiffCond>(my::numscal_)),

      // utility class supporting element evaluation for diffusion-conduction formulation
      utils_(Discret::Elements::ScaTraEleUtilsElchDiffCond<distype>::instance(
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
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::line2>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::line3>;

// 2D elements
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::tri3>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::tri6>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::quad4>;
// template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::quad8>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::quad9>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::nurbs9>;

// 3D elements
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::hex8>;
// template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::hex20>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::hex27>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::tet4>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::tet10>;
// template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::wedge6>;
template class Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::pyramid5>;
// template class
// Discret::Elements::ScaTraEleCalcSTIDiffCond<Core::FE::CellType::nurbs27>;

FOUR_C_NAMESPACE_CLOSE

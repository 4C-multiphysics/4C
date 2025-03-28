// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_vplast_reform_johnsoncook.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"

#include <cmath>
#include <utility>

FOUR_C_NAMESPACE_OPEN


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::PAR::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      strain_rate_prefac_(matdata.parameters.get<double>("STRAIN_RATE_PREFAC")),
      strain_rate_exp_fac_(matdata.parameters.get<double>("STRAIN_RATE_EXP_FAC")),
      init_yield_strength_(matdata.parameters.get<double>("INIT_YIELD_STRENGTH")),
      isotrop_harden_prefac_(matdata.parameters.get<double>("ISOTROP_HARDEN_PREFAC")),
      isotrop_harden_exp_(matdata.parameters.get<double>("ISOTROP_HARDEN_EXP"))
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    Core::Mat::PAR::Parameter* params)
    : Mat::Viscoplastic::Law(params),
      const_pars_(parameter()->strain_rate_pre_fac(), 1.0 / parameter()->strain_rate_exp_fac(),
          parameter()->isotrop_harden_prefac(), parameter()->isotrop_harden_exp(),
          parameter()->init_yield_strength())
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_stress_ratio(
    const double equiv_stress, const double equiv_plastic_strain)
{
  // extract yield strength from the plastic strain and the material parameters
  const double yield_strength =
      (parameter()->init_yield_strength() +
          parameter()->isotrop_harden_prefac() *
              std::pow(equiv_plastic_strain, parameter()->isotrop_harden_exp()));

  return equiv_stress / yield_strength;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const bool log_substep, Mat::ViscoplastErrorType& err_status, const bool update_hist_var)
{
  // first set error status to "no errors"
  err_status = Mat::ViscoplastErrorType::NoErrors;

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    err_status = Mat::ViscoplastErrorType::NegativePlasticStrain;
    return -1;
  }

  // compute the viscoplastic strain rate; first we set it to 0
  double equiv_plastic_strain_rate = 0.0;

  // stress ratio
  double stress_ratio = evaluate_stress_ratio(equiv_stress, equiv_plastic_strain);

  // then we check the yield condition
  if (stress_ratio >= 1.0)
  {
    // compute logarithm \f$ \log (P \exp(E \left[\frac{\overline{\sigma}}{\sigma_{\text{Y}}}
    // - 1.0]) ) \f$
    const double log_temp = const_pars_.log_p + const_pars_.e * (stress_ratio - 1.0);

    // check if characteristic term too large, throw error overflow error if so
    if (((!log_substep) && (std::log(dt) + log_temp > std::log(10.0 + const_pars_.p * dt))) ||
        ((log_substep) && (std::log(dt) + log_temp > std::log(2.0e30 + const_pars_.p * dt))))
    {
      err_status = Mat::ViscoplastErrorType::OverflowError;
      return -1;
    }

    equiv_plastic_strain_rate = std::exp(log_temp) - const_pars_.p;
  }

  return equiv_plastic_strain_rate;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Core::LinAlg::Matrix<2, 1>
Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_derivatives_of_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const bool log_substep, Mat::ViscoplastErrorType& err_status, const bool update_hist_var)
{
  // first set error status to "no errors"
  err_status = Mat::ViscoplastErrorType::NoErrors;

  // used equivalent plastic strain
  double used_equiv_plastic_strain = equiv_plastic_strain;

  // check whether the plastic strain is less than a set value (singularity in the derivatives
  // below)
  if (std::abs(equiv_plastic_strain) < 1.0e-16)
  {
    used_equiv_plastic_strain = 1.0e-16;
  }

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    err_status = Mat::ViscoplastErrorType::NegativePlasticStrain;
    return Core::LinAlg::Matrix<2, 1>{Core::LinAlg::Initialization::zero};
  }

  // extraction of the yield strength from the plastic strain and the material parameters
  const double yield_strength =
      const_pars_.sigma_Y0 + const_pars_.B * std::pow(used_equiv_plastic_strain, const_pars_.N);
  const double log_yield_strength = std::log(yield_strength);
  const double inv_yield_strength = 1.0 / yield_strength;


  // logarithms of equivalent stress and plastic strain
  const double log_equiv_stress = std::log(equiv_stress);
  const double log_equiv_plastic_strain = std::log(used_equiv_plastic_strain);

  // logarithm of the time step
  const double log_dt = std::log(dt);

  // computation of derivatives

  // first we set derivatives to 0
  Core::LinAlg::Matrix<2, 1> equiv_plastic_strain_rate_ders(Core::LinAlg::Initialization::zero);

  // then we check the yield condition
  if (evaluate_stress_ratio(equiv_stress, used_equiv_plastic_strain) >= 1.0)
  {
    // compute first the logarithms of our derivatives (try to avoid overflow!)
    double log_deriv_sigma = const_pars_.log_p_e +
                             const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) -
                             log_yield_strength;
    double log_deriv_eps = const_pars_.log_p_e +
                           const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) +
                           log_equiv_stress - 2.0 * log_yield_strength + const_pars_.log_B_N +
                           (const_pars_.N - 1.0) * log_equiv_plastic_strain;

    // check overflow error using these logarithms
    if ((!log_substep && (log_dt + log_deriv_sigma > 10.0) && (log_dt + log_deriv_eps > 10.0)) &&
        (log_substep && (log_dt + log_deriv_sigma > 2.0e30) && (log_dt + log_deriv_eps > 2.0e30)))
    {
      err_status = Mat::ViscoplastErrorType::OverflowError;
      return Core::LinAlg::Matrix<2, 1>{Core::LinAlg::Initialization::zero};
    }

    // compute the exact derivatives using these logarithms
    equiv_plastic_strain_rate_ders(0, 0) = std::exp(log_deriv_sigma);
    equiv_plastic_strain_rate_ders(1, 0) = -std::exp(log_deriv_eps);
  }

  return equiv_plastic_strain_rate_ders;
}

FOUR_C_NAMESPACE_CLOSE

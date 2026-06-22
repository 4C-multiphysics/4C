// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_vplast_reform_johnsoncook.hpp"

#include "4C_global_data.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mat_inelastic_defgrad_factors_service.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_law.hpp"
#include "4C_utils_exceptions.hpp"

#include <cmath>
#include <utility>


FOUR_C_NAMESPACE_OPEN


using ViscoplastErrorType = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType;

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::PAR::ReformulatedJohnsonCook::ReformulatedJohnsonCook(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      strain_rate_prefac_(matdata.parameters.get<double>("STRAIN_RATE_PREFAC")),
      strain_rate_exp_fac_(matdata.parameters.get<double>("STRAIN_RATE_EXP_FAC")),
      init_yield_strength_(matdata.parameters.get<double>("INIT_YIELD_STRENGTH")),
      isotrop_harden_prefac_(matdata.parameters.get<double>("ISOTROP_HARDEN_PREFAC")),
      isotrop_harden_exp_(matdata.parameters.get<double>("ISOTROP_HARDEN_EXP")),
      ref_temperature_(matdata.parameters.get<double>("REF_TEMPERATURE")),
      melt_temperature_(matdata.parameters.get<double>("MELT_TEMPERATURE")),
      temperature_sens_(matdata.parameters.get<double>("TEMPERATURE_SENS"))
{
  // consistency checks for the temperature
  FOUR_C_ASSERT_ALWAYS(ref_temperature_ > 0.0,
      "Reference temperature must be > 0.0: current value: {}", ref_temperature_);
  FOUR_C_ASSERT_ALWAYS(melt_temperature_ > ref_temperature_,
      "Melting temperature must be > reference temperature: current values: {} (T_m) and {} "
      "(T_ref)",
      melt_temperature_, ref_temperature_);
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
void Mat::Viscoplastic::ReformulatedJohnsonCook::pre_evaluate(
    const Teuchos::ParameterList& params, int gp)
{
  // call pre_evaluate of base class
  Mat::Viscoplastic::Law::pre_evaluate(params, gp);


  // get temperature factors
  double T = parameter()->ref_temperature();
  if (params.isParameter("temperature"))
  {
    T = params.get<double>("temperature");
  }
  const double T_ref = parameter()->ref_temperature();
  const double T_melt = parameter()->melt_temperature();
  const double M = parameter()->temperature_sens();

  // consistency checks for the temperature
  FOUR_C_ASSERT_ALWAYS(T > 0.0 && T < T_melt,
      "Temperature must be > 0.0 and < melting temperature: current value: {} (T) and {} (T_m)", T,
      T_melt);

  // set temperature ratio
  temperature_ratio_ = 1.0;
  log_temperature_ratio_ = 0.0;
  if (T != T_ref)
  {
    FOUR_C_ASSERT_ALWAYS(T_ref < T_melt,
        "You specified the reference temperature {} >= melting temperature: {}! This cannot be "
        "currently resolved by the Reformulated Johnson-Cook law!",
        T_ref, T_melt);
    temperature_ratio_ =
        1.0 - (std::pow(T, M) - std::pow(T_ref, M)) / (std::pow(T_melt, M) - std::pow(T_ref, M));
    log_temperature_ratio_ = std::log(temperature_ratio_);
  }

  // set temperature ratio derivative
  log_neg_temperature_ratio_deriv_ =
      std::log(M * (std::pow(T, M - 1.0)) / (std::pow(T_melt, M) - std::pow(T_ref, M)));
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_stress_ratio(
    const double equiv_stress, const double equiv_plastic_strain)
{
  // extract yield strength from the plastic strain and the material parameters
  InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType yield_strength_err_status{
      InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors};
  const double yield_strength =
      compute_flow_resistance(equiv_stress, equiv_plastic_strain, yield_strength_err_status);
  FOUR_C_ASSERT_ALWAYS(yield_strength_err_status ==
                           InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
      "Something went wrong when evaluating the yield stress: error = {}",
      EnumTools::enum_name(yield_strength_err_status));

  return equiv_stress / yield_strength;
}


/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::compute_flow_resistance(
    const double equiv_stress, const double equiv_plastic_strain,
    Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType& err_status)
{
  // extract yield strength from the plastic strain and the material parameters
  return (parameter()->init_yield_strength() * temperature_ratio_ +
          parameter()->isotrop_harden_prefac() * temperature_ratio_ *
              std::pow(equiv_plastic_strain, parameter()->isotrop_harden_exp()));
}



/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
double Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const double max_plastic_strain_incr, ViscoplastErrorType& err_status,
    const bool update_hist_var)
{
  // first set error status to "no errors"
  err_status = ViscoplastErrorType::no_errors;

  // Check if plastic strain is negative and throw error (handled by the parent material,
  // substepping)
  if (equiv_plastic_strain < 0.0)
  {
    err_status = ViscoplastErrorType::negative_plastic_strain;
    return -1;
  }

  // compute the viscoplastic strain rate; first we set it to 0
  double equiv_plastic_strain_rate = 0.0;

  // save yield strength
  if (update_hist_var)
  {
    // GP index safeguard
    FOUR_C_ASSERT_ALWAYS(
        static_cast<size_t>(gp_) < time_step_quantities_.current_yield_strength_.size(),
        "The current gp index is {} while the stored number of GP within Reformulated Johnson - "
        "Cook is {} ",
        gp_, time_step_quantities_.current_yield_strength_.size());
    InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType yield_strength_err_status =
        InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors;
    time_step_quantities_.current_yield_strength_[gp_] =
        compute_flow_resistance(equiv_stress, equiv_plastic_strain, yield_strength_err_status);
    FOUR_C_ASSERT_ALWAYS(
        yield_strength_err_status ==
            InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
        "Computing yield strength during the evaluation of the plastic strain rate has failed with "
        "error {}",
        EnumTools::enum_name(yield_strength_err_status));
  }

  // verify whether the maximum plastic strain increment is larger than 0, since we will take its
  // logarithm
  FOUR_C_ASSERT_ALWAYS(max_plastic_strain_incr > 0.0,
      "Maximum plastic strain increment must be > 0: current value = {}", max_plastic_strain_incr);

  // stress ratio
  double stress_ratio = evaluate_stress_ratio(equiv_stress, equiv_plastic_strain);

  // then we check the yield condition
  if (stress_ratio >= 1.0)
  {
    // compute logarithm \f$ \log (P \exp(E \left[\frac{\overline{\sigma}}{\sigma_{\text{Y}}}
    // - 1.0]) ) \f$
    const double log_temp = const_pars_.log_p + const_pars_.e * (stress_ratio - 1.0);

    // check if characteristic term too large, throw error overflow
    // error if so
    if (std::log(dt) + log_temp > std::log(max_plastic_strain_incr + const_pars_.p * dt))
    {
      err_status = ViscoplastErrorType::overflow_error;
      return -1;
    }

    equiv_plastic_strain_rate = std::exp(log_temp) - const_pars_.p;
  }

  return equiv_plastic_strain_rate;
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs
Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_derivatives_of_plastic_strain_rate(
    const double equiv_stress, const double equiv_plastic_strain, const double dt,
    const double max_plastic_strain_deriv_incr, ViscoplastErrorType& err_status,
    const bool update_hist_var)
{
  // declare derivatives to be computed
  double deriv_equiv_stress{0.0};
  double deriv_plastic_strain{0.0};
  double deriv_temperature{0.0};

  // first set error status to "no errors"
  err_status = ViscoplastErrorType::no_errors;

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
    err_status = ViscoplastErrorType::negative_plastic_strain;
    return InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs{
        .deriv_equiv_stress = 0.0, .deriv_plastic_strain = 0.0, .deriv_temperature = 0.0};
  }

  // extraction of the yield strength from the plastic strain and the material parameters
  InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType yield_strength_err_status{
      InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors};
  const double yield_strength =
      compute_flow_resistance(equiv_stress, used_equiv_plastic_strain, yield_strength_err_status);
  FOUR_C_ASSERT_ALWAYS(yield_strength_err_status ==
                           InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorType::no_errors,
      "Something went wrong in the computation of the yield strength in the plastic strain rate "
      "derivative evaluation: error {}",
      EnumTools::enum_name(yield_strength_err_status));
  const double log_yield_strength = std::log(yield_strength);
  const double inv_yield_strength = 1.0 / yield_strength;
  const double log_yield_strength_reference_temperature =
      log_yield_strength - log_temperature_ratio_;


  // logarithms of equivalent stress and plastic strain
  const double log_equiv_stress = std::log(equiv_stress);
  const double log_equiv_plastic_strain = std::log(used_equiv_plastic_strain);

  // logarithm of the time step
  const double log_dt = std::log(dt);

  // computation of derivatives

  // then we check the yield condition
  if (evaluate_stress_ratio(equiv_stress, equiv_plastic_strain) >= 1.0)
  {
    // compute first the logarithms of our derivatives (try to avoid overflow!)
    double log_deriv_sigma = const_pars_.log_p_e +
                             const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) -
                             log_yield_strength;

    const double log_neg_d_yield_strength_d_temperature =
        log_yield_strength_reference_temperature + log_neg_temperature_ratio_deriv_;

    const double log_deriv_temperature =
        const_pars_.log_p_e + const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) +
        log_equiv_stress - 2.0 * log_yield_strength + log_neg_d_yield_strength_d_temperature;

    // verify whether the maximum plastic strain derivative increment is larger than 0, since we
    // will take its logarithm
    FOUR_C_ASSERT_ALWAYS(max_plastic_strain_deriv_incr > 0.0,
        "Maximum plastic strain derivative increment must be > 0: current value = {}",
        max_plastic_strain_deriv_incr);

    // perfect plasticity
    if (const_pars_.is_perfect_plasticity)
    {
      // check overflow error using these logarithms
      double log_max_plastic_strain_deriv_value = std::log(max_plastic_strain_deriv_incr);
      if ((log_dt + log_deriv_sigma > log_max_plastic_strain_deriv_value) or
          (log_dt + log_deriv_temperature > log_max_plastic_strain_deriv_value))
      {
        err_status = ViscoplastErrorType::failed_computation_flow_resistance_derivs;
        return InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs{
            .deriv_equiv_stress = 0.0, .deriv_plastic_strain = 0.0, .deriv_temperature = 0.0};
      }

      // compute the exact derivatives using these logarithms
      deriv_equiv_stress = std::exp(log_deriv_sigma);
      deriv_plastic_strain = 0.0;
      deriv_temperature = std::exp(log_deriv_temperature);
    }
    // hardening case
    else
    {
      const double log_neg_deriv_eps =
          const_pars_.log_p_e + const_pars_.e * (equiv_stress * inv_yield_strength - 1.0) +
          log_equiv_stress - 2.0 * log_yield_strength + const_pars_.log_B_N +
          (const_pars_.N - 1.0) * log_equiv_plastic_strain + log_temperature_ratio_;
      // check overflow error using these logarithms
      double log_max_plastic_strain_deriv_value = std::log(max_plastic_strain_deriv_incr);
      if ((log_dt + log_deriv_sigma > log_max_plastic_strain_deriv_value) or
          (log_dt + log_neg_deriv_eps > log_max_plastic_strain_deriv_value) or
          (log_dt + log_deriv_temperature > log_max_plastic_strain_deriv_value))
      {
        err_status = ViscoplastErrorType::failed_computation_flow_resistance_derivs;
        return InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs{
            .deriv_equiv_stress = 0.0, .deriv_plastic_strain = 0.0, .deriv_temperature = 0.0};
      }

      // compute the exact derivatives using these logarithms
      deriv_equiv_stress = std::exp(log_deriv_sigma);
      deriv_plastic_strain = -std::exp(log_neg_deriv_eps);
      deriv_temperature = std::exp(log_deriv_temperature);
    }
  }

  return InelasticDefgradTransvIsotropElastViscoplastUtils::PlasticStrainRateDerivs{
      .deriv_equiv_stress = deriv_equiv_stress,
      .deriv_plastic_strain = deriv_plastic_strain,
      .deriv_temperature = deriv_temperature};
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::setup(const int numgp,
    const Discret::Elements::Fibers& fibers,
    const std::optional<Discret::Elements::CoordinateSystem>& coord_system)
{
  time_step_quantities_.current_yield_strength_.resize(numgp, parameter()->init_yield_strength());
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::register_output_data_names(
    std::unordered_map<std::string, int>& names_and_size) const
{
  names_and_size["yield_strength"] = 1;
}

bool Mat::Viscoplastic::ReformulatedJohnsonCook::evaluate_output_data(
    const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const
{
  if (name == "yield_strength")
  {
    for (int gp = 0; gp < static_cast<int>(time_step_quantities_.current_yield_strength_.size());
        ++gp)
    {
      data(gp, 0) = time_step_quantities_.current_yield_strength_[gp];
    }
    return true;
  }

  return false;
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::pack_viscoplastic_law(
    Core::Communication::PackBuffer& data) const
{
  // pack relevant variables
  if (parameter() != nullptr)
  {
    // we need to pack this current value since it also saves the number
    // of Gauss points
    add_to_pack(data, time_step_quantities_.current_yield_strength_);
  }
}

void Mat::Viscoplastic::ReformulatedJohnsonCook::unpack_viscoplastic_law(
    Core::Communication::UnpackBuffer& buffer)
{
  // NOTE: factory method is called during assign_to_source in the unpack method of the
  // multiplicative split framework --> the param class of the viscoplastic law is created, we only
  // need to unpack the history variables
  if (parameter() != nullptr)
  {
    extract_from_pack(buffer, time_step_quantities_.current_yield_strength_);
  }
}

FOUR_C_NAMESPACE_CLOSE

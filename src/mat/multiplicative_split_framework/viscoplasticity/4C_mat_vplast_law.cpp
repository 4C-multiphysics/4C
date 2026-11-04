// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_vplast_law.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_vplast_reform_johnsoncook.hpp"
#include "4C_utils_enum.hpp"

#include <utility>

FOUR_C_NAMESPACE_OPEN

namespace ViscoplastUtils = Mat::InelasticDefgradTransvIsotropElastViscoplastUtils;

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::Law::Law(Core::Mat::PAR::Parameter* params,
    const InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings&
        error_registration_settings)
    : error_registration_settings_(error_registration_settings), params_(params)
{
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
Mat::Viscoplastic::Law::Law() : error_registration_settings_(), params_(nullptr) {}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::shared_ptr<Mat::Viscoplastic::Law> Mat::Viscoplastic::Law::factory(const int matnum,
    const InelasticDefgradTransvIsotropElastViscoplastUtils::ErrorRegistrationSettings&
        error_registration_settings)
{
  // for the sake of safety
  if (Global::Problem::instance()->materials() == nullptr)
    FOUR_C_THROW("List of materials cannot be accessed in the global problem instance.");

  // another safety check
  if (Global::Problem::instance()->materials()->num() == 0)
    FOUR_C_THROW("List of materials in the global problem instance is empty.");

  // retrieve problem instance to read from
  const int probinst = Global::Problem::instance()->materials()->get_read_from_problem();
  // retrieve validated input line of material ID in question
  auto* curmat = Global::Problem::instance(probinst)->materials()->parameter_by_id(matnum);


  // get material type and call corresponding constructors
  const Core::Materials::MaterialType currentMaterialType = curmat->type();
  switch (currentMaterialType)
  {
    case Core::Materials::mvl_reformulated_Johnson_Cook:
    {
      // get pointer to parameter class
      auto* params = dynamic_cast<Mat::Viscoplastic::PAR::ReformulatedJohnsonCook*>(curmat);

      // return pointer to material
      return std::make_shared<Mat::Viscoplastic::ReformulatedJohnsonCook>(
          params, error_registration_settings);
    }

    default:
      FOUR_C_THROW("cannot deal with type {}", curmat->type());
  }
}

/*--------------------------------------------------------------------*
 *--------------------------------------------------------------------*/
std::optional<double> Mat::Viscoplastic::Law::integrate_plastic_strain(
    const InputIntegratePlasticStrain& input_integrate_plastic_strain)
{
  // auxiliaries
  ViscoplastUtils::PlasticStrainRateDerivs plastic_strain_rate_derivs;

  // set initial estimate as the previous plastic strain
  double plastic_strain = input_integrate_plastic_strain.last_plastic_strain;

  // initialize iteration counter
  unsigned int iter = 0;

  // initialize quantities involved in the hardening integration with dummy values
  double plastic_strain_rate = 1.0e10;
  double deriv_plastic_strain_rate = 1.0e10;
  double residual = 1.0e10;
  double jacobian = 1.0e10;

  // initialize evaluation error status
  ViscoplastUtils::ErrorType err_status{ViscoplastUtils::ErrorType::no_errors};

  // Newton-Raphson loop for hardening integration
  while (iter < input_integrate_plastic_strain.max_iter)
  {
    // increment iterations
    ++iter;

    // compute plastic strain rate from the viscoplasticity law
    plastic_strain_rate = evaluate_plastic_strain_rate(input_integrate_plastic_strain.equiv_stress,
        plastic_strain, input_integrate_plastic_strain.step, err_status, false);
    // return directly when encountering error -> this will be handled by the estimate
    // interpolation procedure
    if (err_status != ViscoplastUtils::ErrorType::no_errors) return std::nullopt;

    // compute residual
    residual = plastic_strain - input_integrate_plastic_strain.last_plastic_strain -
               input_integrate_plastic_strain.step * plastic_strain_rate;

    // return solution if converged
    if (std::abs(residual) <= input_integrate_plastic_strain.residual_tolerance)
    {
      return plastic_strain;
    }

    // compute derivative of the plastic strain rate w.r.t. plastic
    // strain
    plastic_strain_rate_derivs =
        evaluate_derivatives_of_plastic_strain_rate(input_integrate_plastic_strain.equiv_stress,
            plastic_strain, input_integrate_plastic_strain.step, err_status);
    deriv_plastic_strain_rate = plastic_strain_rate_derivs.deriv_plastic_strain;

    // return directly when encountering overflow error -> this will be handled by the estimate
    // interpolation directly
    if (err_status != ViscoplastUtils::ErrorType::no_errors) return std::nullopt;

    // compute jacobian and verify that it is non-zero
    jacobian = 1.0 - input_integrate_plastic_strain.step * deriv_plastic_strain_rate;
    if (std::abs(jacobian) <= 1.0e-12)
    {
      return std::nullopt;
    }

    // update solution
    plastic_strain -= residual / jacobian;
  }

  return std::nullopt;
}

FOUR_C_NAMESPACE_CLOSE

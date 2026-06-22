// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elchsinglemat.hpp"

#include "4C_global_data.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"
#include "4C_utils_function_of_time.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::PAR::ElchSingleMat::ElchSingleMat(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      diffusion_coefficient_(matdata.parameters.get<double>("DIFF_COEF")),
      diffusion_coefficient_concentration_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("DIFF_COEF_CONC_SCALE_FUNCT")),
      diffusion_coefficient_temperature_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("DIFF_COEF_TEMP_SCALE_FUNCT")),
      conductivity_(matdata.parameters.get<double>("COND")),
      conductivity_concentration_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("COND_CONC_SCALE_FUNCT")),
      conductivity_temperature_scaling_funct_num_(
          matdata.parameters.get<std::optional<int>>("COND_TEMP_SCALE_FUNCT"))
{
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime&
Mat::PAR::ElchSingleMat::diffusion_coefficient_concentration_scaling_funct()
{
  FOUR_C_ASSERT(has_diffusion_coefficient_concentration_scaling(),
      "You try to access the function defining the optional concentration scaling of the diffusion "
      "coefficient, but no function number has been set! Check your implementation that the input "
      "file parameter 'DIFF_COEF_CONC_SCALE_FUNCT' is properly set and that you only call this "
      "function if this optional quantity is set.");

  if (!diffusion_coefficient_concentration_scaling_funct_.has_value())
  {
    diffusion_coefficient_concentration_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            diffusion_coefficient_concentration_scaling_funct_num_.value()));
  }
  return diffusion_coefficient_concentration_scaling_funct_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime&
Mat::PAR::ElchSingleMat::diffusion_coefficient_temperature_scaling_funct()
{
  FOUR_C_ASSERT(has_diffusion_coefficient_temperature_scaling(),
      "You try to access the function defining the optional temperature scaling of the diffusion "
      "coefficient, but no function number has been set! Check your implementation that the input "
      "file parameter 'DIFF_COEF_TEMP_SCALE_FUNCT' is properly set and that you only call this "
      "function if this optional quantity is set.");

  if (!diffusion_coefficient_temperature_scaling_funct_.has_value())
  {
    diffusion_coefficient_temperature_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            diffusion_coefficient_temperature_scaling_funct_num_.value()));
  }
  return diffusion_coefficient_temperature_scaling_funct_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime&
Mat::PAR::ElchSingleMat::conductivity_concentration_scaling_funct()
{
  FOUR_C_ASSERT(has_conductivity_concentration_scaling(),
      "You try to access the function defining the optional concentration scaling of the "
      "conductivity, but no function number has been set! Check your implementation that the input "
      "file parameter 'COND_CONC_SCALE_FUNCT' is properly set and that you only call this function "
      "if this optional quantity is set.");

  if (!conductivity_concentration_scaling_funct_.has_value())
  {
    conductivity_concentration_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            conductivity_concentration_scaling_funct_num_.value()));
  }
  return conductivity_concentration_scaling_funct_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
const Core::Utils::FunctionOfTime& Mat::PAR::ElchSingleMat::conductivity_temperature_scaling_funct()
{
  FOUR_C_ASSERT(has_conductivity_temperature_scaling(),
      "You try to access the function defining the optional temperature scaling of the "
      "conductivity, but no function number has been set! Check your implementation that the input "
      "file parameter 'COND_TEMP_SCALE_FUNCT' is properly set and that you only call this function "
      "if this optional quantity is set.");

  if (!conductivity_temperature_scaling_funct_.has_value())
  {
    conductivity_temperature_scaling_funct_.emplace(
        Global::Problem::instance()->function_by_id<Core::Utils::FunctionOfTime>(
            conductivity_temperature_scaling_funct_num_.value()));
  }
  return conductivity_temperature_scaling_funct_->get();
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
Mat::ElchSingleMat::ElchSingleMat(Mat::PAR::ElchSingleMat* params) : params_(params) {}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_diffusion_coefficient(
    const double concentration, const double temperature) const
{
  // D
  double diffusion_coefficient = params_->diffusion_coefficient_;

  // D = D*D(c)
  if (params_->has_diffusion_coefficient_concentration_scaling())
  {
    diffusion_coefficient *=
        params_->diffusion_coefficient_concentration_scaling_funct().evaluate(concentration);
  }
  // D = D*D(c)*D(T)
  if (params_->has_diffusion_coefficient_temperature_scaling())
  {
    diffusion_coefficient *=
        params_->diffusion_coefficient_temperature_scaling_funct().evaluate(temperature);
  }

  return diffusion_coefficient;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_diffusion_coefficient_concentration_dependent(
    const double concentration) const
{
  // D
  double diffusion_coefficient = params_->diffusion_coefficient_;

  // D = D*D(c)
  if (params_->has_diffusion_coefficient_concentration_scaling())
  {
    diffusion_coefficient *=
        params_->diffusion_coefficient_concentration_scaling_funct().evaluate(concentration);
  }

  return diffusion_coefficient;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_concentration_derivative_of_diffusion_coefficient(
    const double concentration, const double temperature) const
{
  if (not params_->has_diffusion_coefficient_concentration_scaling())
  {
    return 0.0;
  }

  // Computation of D*D'(c)
  double diffusion_coeff_conc_deriv =
      params_->diffusion_coefficient_ *
      params_->diffusion_coefficient_concentration_scaling_funct().evaluate_derivative(
          concentration);

  // Computation of D*D'(c)*D(T)
  if (params_->has_diffusion_coefficient_temperature_scaling())
  {
    diffusion_coeff_conc_deriv *=
        params_->diffusion_coefficient_temperature_scaling_funct().evaluate(temperature);
  }

  return diffusion_coeff_conc_deriv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_temperature_derivative_of_diffusion_coefficient(
    const double concentration, const double temperature) const
{
  if (not params_->has_diffusion_coefficient_temperature_scaling())
  {
    return 0.0;
  }

  // Computation of D*D'(T)
  double diffusion_coeff_temp_deriv =
      params_->diffusion_coefficient_ *
      params_->diffusion_coefficient_temperature_scaling_funct().evaluate_derivative(temperature);

  // Computation of D*D'(T)*D(c)
  if (params_->has_diffusion_coefficient_concentration_scaling())
  {
    diffusion_coeff_temp_deriv *=
        params_->diffusion_coefficient_concentration_scaling_funct().evaluate(concentration);
  }

  return diffusion_coeff_temp_deriv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_conductivity(
    const double concentration, const double temperature) const
{
  // cond
  double conductivity = params_->conductivity_;

  // cond = cond*cond(c)
  if (params_->has_conductivity_concentration_scaling())
  {
    conductivity *= params_->conductivity_concentration_scaling_funct().evaluate(concentration);
  }

  // cond = cond*cond(c)*cond(T)
  if (params_->has_conductivity_temperature_scaling())
  {
    conductivity *= params_->conductivity_temperature_scaling_funct().evaluate(temperature);
  }

  return conductivity;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_concentration_derivative_of_conductivity(
    const double concentration, const double temperature) const
{
  if (not params_->has_conductivity_concentration_scaling())
  {
    return 0.0;
  }

  // Computation of cond*cond'(c)
  double conductivity_conc_deriv =
      params_->conductivity_ *
      params_->conductivity_concentration_scaling_funct().evaluate_derivative(concentration);

  // Computation of cond*cond'(c)*cond(T)
  if (params_->has_conductivity_temperature_scaling())
  {
    conductivity_conc_deriv *=
        params_->conductivity_temperature_scaling_funct().evaluate(temperature);
  }

  return conductivity_conc_deriv;
}

/*----------------------------------------------------------------------*
 *----------------------------------------------------------------------*/
double Mat::ElchSingleMat::compute_temperature_derivative_of_conductivity(
    const double concentration, const double temperature) const
{
  if (not params_->has_conductivity_temperature_scaling())
  {
    return 0.0;
  }

  // Computation of cond*cond'(T)
  double conductivity_temp_deriv =
      params_->conductivity_ *
      params_->conductivity_temperature_scaling_funct().evaluate_derivative(temperature);

  // Computation of cond*cond'(T)*cond(c)
  if (params_->has_conductivity_concentration_scaling())
  {
    conductivity_temp_deriv *=
        params_->conductivity_concentration_scaling_funct().evaluate(concentration);
  }

  return conductivity_temp_deriv;
}

FOUR_C_NAMESPACE_CLOSE

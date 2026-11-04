// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_poro_law.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_mat_poro_density_law.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::PAR::PoroLaw::PoroLaw(const Core::Mat::PAR::Parameter::Data& matdata) : Parameter(matdata) {}

Mat::PAR::PoroLawLinear::PoroLawLinear(const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata), bulk_modulus_(matdata.parameters.get<double>("BULKMODULUS"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawLinear::create_material() { return nullptr; }

void Mat::PAR::PoroLawLinear::compute_porosity(const double& refporosity, const double& press,
    const double& J, const int& gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  porosity = refporosity + 1.0 / bulk_modulus_ * press + (1.0 - refporosity) * (J - 1.0);

  if (dphi_dp) *dphi_dp = 1.0 / bulk_modulus_;
  if (dphi_dJ) *dphi_dJ = (1.0 - refporosity);
  if (dphi_dJdp) *dphi_dJdp = 0.0;
  if (dphi_dJJ) *dphi_dJJ = 0.0;
  if (dphi_dpp) *dphi_dpp = 0.0;
  if (dphi_dphiref) *dphi_dphiref = 2.0 - J;
}

void Mat::PAR::PoroLawLinear::constitutive_derivatives(const Teuchos::ParameterList& params,
    const double& press, const double& J, const double& porosity, const double& refporosity,
    double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref, double* W)
{
  if (W) *W = bulk_modulus_ * (porosity - refporosity - (1.0 - refporosity) * (J - 1.0)) - press;
  if (dW_dp) *dW_dp = -1.0;
  if (dW_dphi) *dW_dphi = bulk_modulus_;
  if (dW_dJ) *dW_dJ = -bulk_modulus_ * (1.0 - refporosity);
  if (dW_dphiref) *dW_dphiref = bulk_modulus_ * (-2.0 + J);
}

Mat::PAR::PoroLawNeoHooke::PoroLawNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata),
      bulk_modulus_(matdata.parameters.get<double>("BULKMODULUS")),
      penalty_parameter_(matdata.parameters.get<double>("PENALTYPARAMETER"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawNeoHooke::create_material()
{
  return nullptr;
}

void Mat::PAR::PoroLawNeoHooke::compute_porosity(const double& refporosity, const double& press,
    const double& J, const int& gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  const double& bulkmodulus = bulk_modulus_;
  const double& penalty = penalty_parameter_;

  const double a = (bulkmodulus / (1 - refporosity) + press - penalty / refporosity) * J;
  const double b = -a + bulkmodulus + penalty;
  const double c = b * b + 4.0 * penalty * a;
  double d = sqrt(c);


  double test = 1 / (2.0 * a) * (-b + d);
  double sign = 1.0;
  if (test >= 1.0 or test < 0.0)
  {
    sign = -1.0;
    d = sign * d;
  }

  const double a_inv = 1.0 / a;
  const double d_inv = 1.0 / d;
  const double J_inv = 1.0 / J;

  const double phi = 1 / (2 * a) * (-b + d);


  const double d_p = J * (-b + 2.0 * penalty) * d_inv;
  const double d_p_p = (d * J + d_p * (b - 2.0 * penalty)) * d_inv * d_inv * J;
  const double d_J = a * J_inv * (-b + 2.0 * penalty) * d_inv;
  const double d_J_p = (d_p * J_inv + (1 - d_p * d_p * J_inv * J_inv) * d_inv * a);
  const double d_J_J = (a * a * J_inv * J_inv - d_J * d_J) * d_inv;

  // d(porosity) / d(p)
  if (dphi_dp) *dphi_dp = (-J * phi + 0.5 * (J + d_p)) * a_inv;

  // d(porosity) / d(J)
  if (dphi_dJ) *dphi_dJ = (-phi + 0.5) * J_inv + 0.5 * d_J * a_inv;

  // d(porosity) / d(J)d(pressure)
  if (dphi_dJdp and dphi_dp)
    *dphi_dJdp = -J_inv * (*dphi_dp) + 0.5 * d_J_p * a_inv - 0.5 * d_J * J * a_inv * a_inv;

  // d^2(porosity) / d(J)^2
  if (dphi_dJJ)
    *dphi_dJJ = phi * J_inv * J_inv - (*dphi_dJ) * J_inv - 0.5 * J_inv * J_inv -
                0.5 * d_J * J_inv * a_inv + 0.5 * d_J_J * a_inv;

  // d^2(porosity) / d(pressure)^2
  if (dphi_dpp and dphi_dp)
    *dphi_dpp = -J * a_inv * (*dphi_dp) + phi * J * J * a_inv * a_inv -
                0.5 * J * a_inv * a_inv * (J + d_p) + 0.5 * d_p_p * a_inv;

  porosity = phi;

  if (dphi_dphiref)
  {
    const double dadphiref = J * (bulkmodulus / ((1 - refporosity) * (1 - refporosity)) +
                                     penalty / (refporosity * refporosity));
    const double tmp = 2 * dadphiref * a_inv * (-b * (a + b) * a_inv - 2 * penalty);
    const double dddphiref = sign * (dadphiref * sqrt(c) * a_inv + tmp);

    *dphi_dphiref = (a * (dadphiref + dddphiref) - dadphiref * (-b + d)) * 0.5 * a_inv * a_inv;
  }
}

void Mat::PAR::PoroLawNeoHooke::constitutive_derivatives(const Teuchos::ParameterList& params,
    const double& press, const double& J, const double& porosity, const double& refporosity,
    double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref, double* W)
{
  // some intermediate values
  const double a = bulk_modulus_ / (1 - refporosity) + press - penalty_parameter_ / refporosity;
  const double b = -1.0 * J * a + bulk_modulus_ + penalty_parameter_;

  const double scale = 1.0 / bulk_modulus_;

  // scale everything with 1/bulkmodulus (I hope this will help the solver...)
  if (W) *W = (J * a * porosity * porosity + porosity * b - penalty_parameter_) * scale;
  if (dW_dp) *dW_dp = (-1.0 * J * porosity * (1.0 - porosity)) * scale;
  if (dW_dphi) *dW_dphi = (2.0 * J * a * porosity + b) * scale;
  if (dW_dJ) *dW_dJ = (a * porosity * porosity - porosity * a) * scale;

  if (dW_dphiref)
  {
    const double dadphiref = J * (bulk_modulus_ / ((1 - refporosity) * (1 - refporosity)) +
                                     penalty_parameter_ / (refporosity * refporosity));
    const double dbdphiref = -1.0 * J * dadphiref;

    *dW_dphiref = (J * dadphiref * porosity * porosity + porosity * dbdphiref) * scale;
  }
}

Mat::PAR::PoroLawConstant::PoroLawConstant(const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawConstant::create_material()
{
  return nullptr;
}

void Mat::PAR::PoroLawConstant::compute_porosity(const double& refporosity, const double& press,
    const double& J, const int& gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  // porosity is constant -> derivatives are zero
  porosity = refporosity;
  if (dphi_dp) *dphi_dp = 0.0;
  if (dphi_dJ) *dphi_dJ = 0.0;
  if (dphi_dJdp) *dphi_dJdp = 0.0;
  if (dphi_dJJ) *dphi_dJJ = 0.0;
  if (dphi_dpp) *dphi_dpp = 0.0;
  if (dphi_dphiref) *dphi_dphiref = 1.0;
}

void Mat::PAR::PoroLawConstant::constitutive_derivatives(const Teuchos::ParameterList& params,
    const double& press, const double& J, const double& porosity, const double& refporosity,
    double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref, double* W)
{
  if (W) *W = porosity - refporosity;
  if (dW_dp) *dW_dp = 0.0;
  if (dW_dphi) *dW_dphi = 1.0;
  if (dW_dJ) *dW_dJ = 0.0;
  if (dW_dphiref) *dW_dphiref = -1.0;
}

Mat::PAR::PoroLawIncompSkeleton::PoroLawIncompSkeleton(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata)
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawIncompSkeleton::create_material()
{
  return nullptr;
}

void Mat::PAR::PoroLawIncompSkeleton::compute_porosity(const double& refporosity,
    const double& press, const double& J, const int& gp, double& porosity, double* dphi_dp,
    double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  porosity = 1.0 - (1.0 - refporosity) / J;

  if (dphi_dp) *dphi_dp = 0.0;
  if (dphi_dJ) *dphi_dJ = (1.0 - refporosity) / (J * J);
  if (dphi_dJdp) *dphi_dJdp = 0.0;
  if (dphi_dJJ) *dphi_dJJ = -2.0 * (1.0 - refporosity) / (J * J * J);
  if (dphi_dpp) *dphi_dpp = 0.0;
  if (dphi_dphiref) *dphi_dphiref = 1.0 / J;
}

void Mat::PAR::PoroLawIncompSkeleton::constitutive_derivatives(const Teuchos::ParameterList& params,
    const double& press, const double& J, const double& porosity, const double& refporosity,
    double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref, double* W)
{
  if (W) *W = J * (1.0 - porosity) - (1.0 - refporosity);
  if (dW_dp) *dW_dp = 0.0;
  if (dW_dphi) *dW_dphi = -1.0 * J;
  if (dW_dJ) *dW_dJ = 1.0 - porosity;
  if (dW_dphiref) *dW_dphiref = 1.0;
}

Mat::PAR::PoroLawLinBiot::PoroLawLinBiot(const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata),
      inv_biot_modulus_(matdata.parameters.get<double>("INVBIOTMODULUS")),
      biot_coeff_(matdata.parameters.get<double>("BIOTCEOFF"))
{
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawLinBiot::create_material() { return nullptr; }

void Mat::PAR::PoroLawLinBiot::compute_porosity(const double& refporosity, const double& press,
    const double& J, const int& gp, double& porosity, double* dphi_dp, double* dphi_dJ,
    double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  porosity = refporosity + inv_biot_modulus_ * press + biot_coeff_ * (J - 1);

  if (dphi_dp) *dphi_dp = inv_biot_modulus_;
  if (dphi_dJ) *dphi_dJ = biot_coeff_;
  if (dphi_dJdp) *dphi_dJdp = 0.0;
  if (dphi_dJJ) *dphi_dJJ = 0.0;
  if (dphi_dpp) *dphi_dpp = 0.0;
  if (dphi_dphiref) *dphi_dphiref = 1.0;
}

void Mat::PAR::PoroLawLinBiot::constitutive_derivatives(const Teuchos::ParameterList& params,
    const double& press, const double& J, const double& porosity, const double& refporosity,
    double* dW_dp, double* dW_dphi, double* dW_dJ, double* dW_dphiref, double* W)
{
  if (W) *W = porosity - refporosity - inv_biot_modulus_ * press - biot_coeff_ * (J - 1);
  if (dW_dp) *dW_dp = -1.0 * inv_biot_modulus_;
  if (dW_dphi) *dW_dphi = 1.0;
  if (dW_dJ) *dW_dJ = -1.0 * biot_coeff_;
  if (dW_dphiref) *dW_dphiref = -1.0;
}

Mat::PAR::PoroLawDensityDependent::PoroLawDensityDependent(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : PoroLaw(matdata)
{
  const int densityID = matdata.parameters.get<int>("DENSITYLAWID");
  density_law_ = Mat::PAR::PoroDensityLaw::create_density_law(densityID);
}

std::shared_ptr<Core::Mat::Material> Mat::PAR::PoroLawDensityDependent::create_material()
{
  return nullptr;
}

void Mat::PAR::PoroLawDensityDependent::compute_porosity(const double& refporosity,
    const double& press, const double& J, const int& gp, double& porosity, double* dphi_dp,
    double* dphi_dJ, double* dphi_dJdp, double* dphi_dJJ, double* dphi_dpp, double* dphi_dphiref)
{
  // compute relation of reference to current density
  const double reldensity = density_law_->compute_ref_density_to_cur_density(press);
  const double reldensityderiv = density_law_->compute_ref_density_to_cur_density_derivative(press);
  const double reldensityderivderiv =
      density_law_->compute_ref_density_to_cur_density_second_derivative(press);

  // compute porosity
  porosity = 1.0 - reldensity * (1.0 - refporosity) / J;

  if (dphi_dp) *dphi_dp = -1.0 * reldensityderiv * (1.0 - refporosity) / J;
  if (dphi_dJ) *dphi_dJ = reldensity * (1.0 - refporosity) / (J * J);
  if (dphi_dJdp) *dphi_dJdp = reldensityderiv * (1.0 - refporosity) / (J * J);
  if (dphi_dJJ) *dphi_dJJ = -2.0 * reldensityderiv * (1.0 - refporosity) / (J * J * J);
  if (dphi_dpp) *dphi_dpp = -1.0 * reldensityderivderiv * (1.0 - refporosity) / J;
  if (dphi_dphiref) *dphi_dphiref = reldensity / J;
}

void Mat::PAR::PoroLawDensityDependent::constitutive_derivatives(
    const Teuchos::ParameterList& params, const double& press, const double& J,
    const double& porosity, const double& refporosity, double* dW_dp, double* dW_dphi,
    double* dW_dJ, double* dW_dphiref, double* W)
{
  // compute relation of reference to current density
  const double reldensity = density_law_->compute_ref_density_to_cur_density(press);
  const double reldensityderiv = density_law_->compute_ref_density_to_cur_density_derivative(press);

  if (W) *W = porosity - 1.0 + reldensity * (1.0 - refporosity) / J;
  if (dW_dp) *dW_dp = reldensityderiv * (1.0 - refporosity) / J;
  if (dW_dphi) *dW_dphi = 1.0;
  if (dW_dJ) *dW_dJ = -1.0 * reldensity * (1.0 - refporosity) / (J * J);
  if (dW_dphiref) *dW_dphiref = -1.0 * reldensity / J;
}

double Mat::PAR::PoroLawDensityDependent::inv_bulk_modulus() const
{
  return density_law_->inv_bulkmodulus();
}

FOUR_C_NAMESPACE_CLOSE

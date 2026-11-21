// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupneohooke.hpp"

#include "4C_material_parameter_base.hpp"

#include <limits>

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupNeoHooke::CoupNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      youngs_(matdata.parameters.get<double>("YOUNG")),
      nue_(matdata.parameters.get<double>("NUE"))
{
  // Material Constants c and beta
  c_ = youngs_ / (4.0 * (1.0 + nue_));
  beta_ = nue_ / (1.0 - 2.0 * nue_);
}

Mat::Elastic::CoupNeoHooke::CoupNeoHooke(Mat::Elastic::PAR::CoupNeoHooke* params) : params_(params)
{
}

void Mat::Elastic::CoupNeoHooke::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constants c and beta
  const double c = params_->c_;
  const double beta = params_->beta_;

  // strain energy: psi = c / beta * (I3^{-beta} - 1) + c * (I1 - 3)
  double psiadd = c * (prinv(0) - 3.);
  if (beta != 0)  // take care of possible division by zero in case or Poisson's ratio nu = 0.0
    psiadd += (c / beta) * (std::exp(std::log(prinv(2)) * (-beta)) - 1.);
  else
    psiadd -= c * std::log(prinv(2));

  // add to overall strain energy
  psi += psiadd;
}

void Mat::Elastic::CoupNeoHooke::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double beta = params_->beta_;
  const double c = params_->c_;

  dPI(0) += c;
  // computing exp(log(a)*b) is faster than pow(a,b)
  if (prinv(2) > 0)
  {
    const double prinv2_to_beta_m1 = std::exp(std::log(prinv(2)) * (-beta - 1.));
    dPI(2) -= c * prinv2_to_beta_m1;
    ddPII(2) += c * (beta + 1.) * prinv2_to_beta_m1 / prinv(2);
  }
  else
    dPI(2) = ddPII(2) = std::numeric_limits<double>::quiet_NaN();
}

void Mat::Elastic::CoupNeoHooke::add_third_derivatives_principal_iso(
    Core::LinAlg::Matrix<10, 1>& dddPIII_iso, const Core::LinAlg::Matrix<3, 1>& prinv_iso,
    const int gp, const int eleGID)
{
  const double beta = params_->beta_;
  const double c = params_->c_;

  dddPIII_iso(2) -= c * (beta + 1.0) * (beta + 2.0) * std::pow(prinv_iso(2), -beta - 3.0);
}

void Mat::Elastic::CoupNeoHooke::add_coup_deriv_vol(
    const double J, double* dPj1, double* dPj2, double* dPj3, double* dPj4)
{
  const double beta = params_->beta_;
  const double c = params_->c_;

  if (J < 0.) FOUR_C_THROW("negative jacobian determinant");

  if (dPj1) *dPj1 += 2. * c * pow(J, -1. / 3.) - 2. * c * pow(J * J, -beta) / J;
  if (dPj2)
    *dPj2 += -2. / 3. * c * pow(J, -4. / 3.) + 4. * c * pow(J * J, -beta) * beta * pow(J, -2.) +
             2. * c * pow(J * J, -beta) * pow(J, -2.);
  if (dPj3)
    *dPj3 += 0.8e1 / 0.9e1 * c * pow(J, -0.7e1 / 3.) -
             0.8e1 * c * pow(J * J, -beta) * beta * beta * pow(J, -3.) -
             0.12e2 * c * pow(J * J, -beta) * beta * pow(J, -3.) -
             4. * c * pow(J * J, -beta) * pow(J, -3.);
  if (dPj4)
    *dPj4 += -56. / 27. * c * pow(J, -10. / 3.) +
             16. * c * pow(J * J, -beta) * pow(beta, 3.) * pow(J, -4.) +
             48. * c * pow(J * J, -beta) * beta * beta * pow(J, -4.) +
             44. * c * pow(J * J, -beta) * beta * pow(J, -4.) +
             12. * c * pow(J * J, -beta) * pow(J, -4.);
}
FOUR_C_NAMESPACE_CLOSE

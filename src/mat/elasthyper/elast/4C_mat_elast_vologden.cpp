// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_vologden.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::VolOgden::VolOgden(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      kappa_(matdata.parameters.get<double>("KAPPA")),
      beta_(matdata.parameters.get<double>("BETA"))
{
}

Mat::Elastic::VolOgden::VolOgden(Mat::Elastic::PAR::VolOgden* params) : params_(params) {}

void Mat::Elastic::VolOgden::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double kappa = params_->kappa_;
  const double beta = params_->beta_;

  // strain energy: Psi = \frac {\kappa}{\beta^2}(\beta lnJ + J^{-\beta}-1)
  // add to overall strain energy
  if (beta != 0)
    psi += kappa / (beta * beta) * (beta * log(modinv(2)) + pow(modinv(2), -beta) - 1.);
  else
    psi += kappa / 2. * pow(std::log(modinv(2)), 2.);
}

void Mat::Elastic::VolOgden::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double kappa = params_->kappa_;
  const double beta = params_->beta_;

  if (beta != 0.)
  {
    dPmodI(2) += kappa / (beta * beta) * (beta / modinv(2) - beta * pow(modinv(2), -beta - 1.));
    ddPmodII(2) +=
        kappa / (beta * beta) *
        (-beta / (modinv(2) * modinv(2)) + beta * (beta + 1.) * pow(modinv(2), -beta - 2.));
  }
  else
  {
    dPmodI(2) += kappa * std::log(modinv(2)) / modinv(2);
    ddPmodII(2) += kappa * (1. - std::log(modinv(2))) / (modinv(2) * modinv(2));
  }
}

void Mat::Elastic::VolOgden::add3rd_vol_deriv(
    const Core::LinAlg::Matrix<3, 1>& modinv, double& d3PsiVolDJ3)
{
  const double kappa = params_->kappa_;
  const double beta = params_->beta_;
  const double J = modinv(2);
  if (beta != 0.)
    d3PsiVolDJ3 +=
        kappa / (beta * J * J * J) * (2. - pow(J, -beta) * (beta * beta + 3. * beta + 2.));
  else
    d3PsiVolDJ3 += kappa * (-3. + 2. * std::log(modinv(2))) / (modinv(2) * modinv(2) * modinv(2));
}
FOUR_C_NAMESPACE_CLOSE

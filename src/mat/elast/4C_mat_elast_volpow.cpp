// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_volpow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::VolPow::VolPow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      a_(matdata.parameters.get<double>("A")),
      expon_(matdata.parameters.get<double>("EXPON"))
{
}

Mat::Elastic::VolPow::VolPow(Mat::Elastic::PAR::VolPow* params) : params_(params) {}

void Mat::Elastic::VolPow::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double a = params_->a_;
  const double expon = params_->expon_;

  // strain energy: Psi = \frac{a}{expon-1} J^(-expon+1) + aJ
  // add to overall strain energy
  psi += a / (expon - 1.) * std::pow(modinv(2), -expon + 1.) + a * modinv(2);
}

void Mat::Elastic::VolPow::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double a = params_->a_;
  const double expon = params_->expon_;

  dPmodI(2) += -a * (std::pow(modinv(2), -expon) - 1.);

  ddPmodII(2) += expon * a * std::pow(modinv(2), -(expon + 1.));
}

void Mat::Elastic::VolPow::add3rd_vol_deriv(
    const Core::LinAlg::Matrix<3, 1>& modinv, double& d3PsiVolDJ3)
{
  const double a = params_->a_;
  const double expon = params_->expon_;

  d3PsiVolDJ3 += -expon * a * (expon + 1.) * std::pow(modinv(2), -(expon + 2.));
}
FOUR_C_NAMESPACE_CLOSE

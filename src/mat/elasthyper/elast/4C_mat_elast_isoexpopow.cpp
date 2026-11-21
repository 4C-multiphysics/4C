// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isoexpopow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoExpoPow::IsoExpoPow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      k1_(matdata.parameters.get<double>("K1")),
      k2_(matdata.parameters.get<double>("K2")),
      d_(matdata.parameters.get<int>("C"))
{
}

Mat::Elastic::IsoExpoPow::IsoExpoPow(Mat::Elastic::PAR::IsoExpoPow* params) : params_(params) {}

void Mat::Elastic::IsoExpoPow::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double k1 = params_->k1_;
  const double k2 = params_->k2_;
  const int d = params_->d_;

  // strain energy: Psi = \frac{k_1}{2k_2} (e^{k_2 (\overline{I}_{\boldsymbol{C}}-3)^d}-1).
  // add to overall strain energy
  if (k2 != 0) psi += k1 / (2. * k2) * (exp(k2 * pow(modinv(0) - 3., d)) - 1.);
}

void Mat::Elastic::IsoExpoPow::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double k1 = params_->k1_;
  const double k2 = params_->k2_;
  const int d = params_->d_;
  const double expf = std::exp(k2 * std::pow(modinv(0) - 3., (double)d));

  if (d < 1)
    FOUR_C_THROW(
        "The Elast_IsoExpoPow - material only works for positive integer exponents (C) larger than "
        "one.");

  if (d == 1)
    dPmodI(0) += k1 * d * 0.5 * expf;
  else
    dPmodI(0) += k1 * d * 0.5 * expf * pow(modinv(0) - 3., d - 1.);

  if (d == 1)
    ddPmodII(0) += k1 * d * d * k2 * 0.5 * expf;
  else if (d == 2)
    ddPmodII(0) += k1 * d * d * k2 * 0.5 * expf * pow((modinv(0) - 3.), 2. * (d - 1.)) +
                   k1 * d * 0.5 * expf * (d - 1.);
  else
    ddPmodII(0) += k1 * d * d * k2 * 0.5 * expf * pow((modinv(0) - 3.), 2. * (d - 1.)) +
                   k1 * d * 0.5 * expf * (d - 1.) * pow((modinv(0) - 3.), (d - 2.));
}
FOUR_C_NAMESPACE_CLOSE

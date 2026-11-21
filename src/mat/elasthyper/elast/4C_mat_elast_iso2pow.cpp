// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_iso2pow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::Iso2Pow::Iso2Pow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c_(matdata.parameters.get<double>("C")),
      d_(matdata.parameters.get<int>("D"))
{
}

Mat::Elastic::Iso2Pow::Iso2Pow(Mat::Elastic::PAR::Iso2Pow* params) : params_(params) {}

void Mat::Elastic::Iso2Pow::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constants c and d
  const double c = params_->c_;
  const int d = params_->d_;

  // strain energy: Psi = C (\overline{II}_{\boldsymbol{C}}-3)^D
  // add to overall strain energy
  psi += c * pow((modinv(1) - 3.), d);
}

void Mat::Elastic::Iso2Pow::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double c = params_->c_;
  const int d = params_->d_;

  if (d < 1)
    FOUR_C_THROW(
        "The Elast_Iso2Pow - material only works for positive integer exponents larger than one.");

  if (d == 1)
    dPmodI(1) += c * d;
  else
    dPmodI(1) += c * d * pow(modinv(1) - 3., d - 1.);

  if (d == 1)
    ddPmodII(1) += 0.;
  else if (d == 2)
    ddPmodII(1) += c * d * (d - 1.);
  else
    ddPmodII(1) += c * d * (d - 1.) * pow(modinv(1) - 3., d - 2.);
}
FOUR_C_NAMESPACE_CLOSE

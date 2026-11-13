// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isoyeoh.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoYeoh::IsoYeoh(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c1_(matdata.parameters.get<double>("C1")),
      c2_(matdata.parameters.get<double>("C2")),
      c3_(matdata.parameters.get<double>("C3"))
{
}

Mat::Elastic::IsoYeoh::IsoYeoh(Mat::Elastic::PAR::IsoYeoh* params) : params_(params) {}

void Mat::Elastic::IsoYeoh::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;
  const double c3 = params_->c3_;

  // strain energy: Psi = C1 (\overline{I}_{\boldsymbol{C}}-3) + C2
  // (\overline{I}_{\boldsymbol{C}}-3)^2 + C3 (\overline{I}_{\boldsymbol{C}}-3)^3. add to overall
  // strain energy
  psi += c1 * (modinv(0) - 3.) + c2 * (modinv(0) - 3.) * (modinv(0) - 3.) +
         c3 * (modinv(0) - 3.) * (modinv(0) - 3.) * (modinv(0) - 3.);
}

void Mat::Elastic::IsoYeoh::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;
  const double c3 = params_->c3_;

  dPmodI(0) += c1 + 2. * c2 * (modinv(0) - 3.) + 3. * c3 * (modinv(0) - 3.) * (modinv(0) - 3.);
  ddPmodII(0) += 2. * c2 + 6. * c3 * (modinv(0) - 3.);
}
FOUR_C_NAMESPACE_CLOSE

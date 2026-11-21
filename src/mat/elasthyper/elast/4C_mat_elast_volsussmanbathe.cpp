// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_volsussmanbathe.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::VolSussmanBathe::VolSussmanBathe(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), kappa_(matdata.parameters.get<double>("KAPPA"))
{
}

Mat::Elastic::VolSussmanBathe::VolSussmanBathe(Mat::Elastic::PAR::VolSussmanBathe* params)
    : params_(params)
{
}

void Mat::Elastic::VolSussmanBathe::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double kappa = params_->kappa_;

  // strain energy: Psi = \frac \kappa 2 (J-1)^2
  // add to overall strain energy
  psi += kappa * 0.5 * (modinv(2) - 1.) * (modinv(2) - 1.);
}

void Mat::Elastic::VolSussmanBathe::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double kappa = params_->kappa_;

  dPmodI(2) += kappa * (modinv(2) - 1.);

  ddPmodII(2) += kappa;
}

void Mat::Elastic::VolSussmanBathe::add3rd_vol_deriv(
    const Core::LinAlg::Matrix<3, 1>& modinv, double& d3PsiVolDJ3)
{
  d3PsiVolDJ3 += 0.;
}
FOUR_C_NAMESPACE_CLOSE

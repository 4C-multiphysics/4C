// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coup2pow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::Coup2Pow::Coup2Pow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c_(matdata.parameters.get<double>("C")),
      d_(matdata.parameters.get<int>("D"))
{
}

Mat::Elastic::Coup2Pow::Coup2Pow(Mat::Elastic::PAR::Coup2Pow* params) : params_(params) {}

void Mat::Elastic::Coup2Pow::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constants c and beta
  const double c = params_->c_;
  const int d = params_->d_;

  // strain energy: Psi = C (II_{\boldsymbol{C}}-3)^D
  // add to overall strain energy
  psi += c * pow((prinv(1) - 3.), d);
}

void Mat::Elastic::Coup2Pow::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double c = params_->c_;
  const int d = params_->d_;

  // If d<2 the material model is not stress free in the reference configuration
  if (d < 2)
    FOUR_C_THROW(
        "The Elast_Coup2Pow - material only works for positive integer exponents, which are larger "
        "than two.");

  dPI(1) += c * d * pow((prinv(1) - 3.), d - 1.);

  if (d == 2)
    ddPII(1) += (c * d * d - c * d);
  else
    ddPII(1) += (c * d * d - c * d) * pow((prinv(1) - 3.), d - 2.);
}
FOUR_C_NAMESPACE_CLOSE

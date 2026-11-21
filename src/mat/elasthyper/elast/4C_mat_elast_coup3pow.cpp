// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coup3pow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::Coup3Pow::Coup3Pow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c_(matdata.parameters.get<double>("C")),
      d_(matdata.parameters.get<int>("D"))
{
}

Mat::Elastic::Coup3Pow::Coup3Pow(Mat::Elastic::PAR::Coup3Pow* params) : params_(params) {}

void Mat::Elastic::Coup3Pow::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constants c and beta
  const double c = params_->c_;
  const int d = params_->d_;

  // add to overall strain energy
  psi += c * pow((pow(prinv(2), 1. / 3.) - 1.), d);
}

void Mat::Elastic::Coup3Pow::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double c = params_->c_;
  const int d = params_->d_;

  // If d<2
  if (d < 2)
    FOUR_C_THROW(
        "The Elast_Coup3Pow - material only works for positive integer exponents, which are larger "
        "than two.");

  dPI(2) += 1. / 3. * c * d * pow(prinv(2), -2. / 3.) * pow((pow(prinv(2), 1. / 3.) - 1.), d - 1.);

  if (d == 2)
    ddPII(2) +=
        -2. / 9. * c * d * pow(prinv(2), -5. / 3.) * pow((pow(prinv(2), 1. / 3.) - 1.), d - 1.) +
        1. / 9. * c * d * (d - 1.) * pow(prinv(2), -4. / 3.);
  else
    ddPII(2) +=
        -2. / 9. * c * d * pow(prinv(2), -5. / 3.) * pow((pow(prinv(2), 1. / 3.) - 1.), d - 1.) +
        1. / 9. * c * d * (d - 1.) * pow(prinv(2), -4. / 3.) *
            pow((pow(prinv(2), 1. / 3.) - 1.), d - 2.);
}
FOUR_C_NAMESPACE_CLOSE

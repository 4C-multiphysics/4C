// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupSaintVenantKirchhoff.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupSVK::CoupSVK(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  double c1 = matdata.parameters.get<double>("YOUNG");
  double c2 = matdata.parameters.get<double>("NUE");

  if (c2 <= 0.5 and c2 > -1.0)
  {
    lambda_ = (c2 == 0.5) ? 0.0 : c1 * c2 / ((1.0 + c2) * (1.0 - 2.0 * c2));
    mue_ = c1 / (2.0 * (1.0 + c2));  // shear modulus
  }
  else
    FOUR_C_THROW("Poisson's ratio must be between -1.0 and 0.5!");
}

Mat::Elastic::CoupSVK::CoupSVK(Mat::Elastic::PAR::CoupSVK* params) : params_(params) {}

void Mat::Elastic::CoupSVK::add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
    const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double lambda = params_->lambda_;
  const double mue = params_->mue_;

  // strain energy: Psi = (1/4*mue+1/8*lambda)*I_1^2 - (0.75*lambda+0.5*mue)*I_1 - 0.5*mue*I_2 +
  // 9/8*lambda + 0.75*mue add to overall strain energy
  psi += (0.25 * mue + 0.125 * lambda) * prinv(0) * prinv(0) -
         (0.75 * lambda + 0.5 * mue) * prinv(0) - 0.5 * mue * prinv(1) + 1.125 * lambda +
         0.75 * mue;
}

void Mat::Elastic::CoupSVK::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double lambda = params_->lambda_;
  const double mue = params_->mue_;

  dPI(0) += (0.5 * mue + 0.25 * lambda) * prinv(0) - 0.75 * lambda - 0.5 * mue;
  dPI(1) -= 0.5 * mue;

  ddPII(0) += 0.5 * mue + 0.25 * lambda;
}

void Mat::Elastic::CoupSVK::add_third_derivatives_principal_iso(
    Core::LinAlg::Matrix<10, 1>& dddPIII_iso, const Core::LinAlg::Matrix<3, 1>& prinv_iso,
    const int gp, const int eleGID)
{
  // do nothing
}

void Mat::Elastic::CoupSVK::add_coup_deriv_vol(
    const double J, double* dPj1, double* dPj2, double* dPj3, double* dPj4)
{
  const double lambda = params_->lambda_;
  const double mu = params_->mue_;

  if (dPj1)
    *dPj1 += 12. * (mu / 4. + lambda / 8.) * pow(J, 1. / 3.) -
             2. * (3. / 4. * lambda + mu / 2.) * pow(J, -1. / 3.) - 2. * mu * pow(J, 1. / 3.);
  if (dPj2)
    *dPj2 += 4. * (mu / 4. + lambda / 8.) * pow(J, -2. / 3.) +
             2. / 3. * (3. / 4. * lambda + mu / 2.) * pow(J, -4. / 3.) -
             2. / 3. * mu * pow(J, -2. / 3.);
  if (dPj3)
    *dPj3 += -8. / 3. * (mu / 4. + lambda / 8.) * pow(J, -5. / 3.) -
             8. / 9. * (3. / 4. * lambda + mu / 2.) * pow(J, -7. / 3.) +
             4. / 9. * mu * pow(J, -5. / 3.);
  if (dPj4)
    *dPj4 += 40. / 9. * (mu / 4. + lambda / 8.) * pow(J, -8. / 3.) +
             56. / 27. * (3. / 4. * lambda + mu / 2.) * pow(J, -10. / 3.) -
             20. / 27. * mu * pow(J, -8. / 3.);
}
FOUR_C_NAMESPACE_CLOSE

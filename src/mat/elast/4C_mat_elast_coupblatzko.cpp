// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupblatzko.hpp"

#include "4C_material_parameter_base.hpp"

#include <cmath>

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupBlatzKo::CoupBlatzKo(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      mue_(matdata.parameters.get<double>("MUE")),
      nue_(matdata.parameters.get<double>("NUE")),
      f_(matdata.parameters.get<double>("F"))
{
}

Mat::Elastic::CoupBlatzKo::CoupBlatzKo(Mat::Elastic::PAR::CoupBlatzKo* params) : params_(params) {}

void Mat::Elastic::CoupBlatzKo::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material parameters for isochoric part
  const double mue = params_->mue_;  // Shear modulus
  const double nue = params_->nue_;  // Poisson's ratio
  const double f = params_->f_;      // interpolation parameter

  // introducing beta for consistency with Holzapfel and simplification
  const double beta = nue / (1. - 2. * nue);

  // strain energy: Psi= f \frac {\mu} 2 \left[ (I_{\boldsymbol C}-3)+\frac 1
  //         {\beta} ( III_{\boldsymbol C}^{-\beta} -1) \right]
  //         +(1-f) \frac {\mu} 2 \left[\left( \frac {II_{\boldsymbol
  //         C}}{III_{\boldsymbol C}}-3 \right) + \frac 1 {\beta}
  //         (III_{\boldsymbol C}^{\beta}-1)\right]

  const double log_i3 = std::log(prinv(2));
  const double negative_power_term = beta == 0.0 ? -log_i3 : std::expm1(-beta * log_i3) / beta;
  const double positive_power_term = beta == 0.0 ? log_i3 : std::expm1(beta * log_i3) / beta;

  psi += 0.5 * mue *
         (f * (prinv(0) - 3.0 + negative_power_term) +
             (1.0 - f) * (prinv(1) / prinv(2) - 3.0 + positive_power_term));
}

void Mat::Elastic::CoupBlatzKo::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  // material parameters for isochoric part
  const double mue = params_->mue_;  // Shear modulus
  const double nue = params_->nue_;  // Poisson's ratio
  const double f = params_->f_;      // interpolation parameter

  // introducing beta for consistency with Holzapfel and simplification
  const double beta = nue / (1. - 2. * nue);


  dPI(0) += f * mue / 2.;
  dPI(1) += (1. - f) * mue / (2. * prinv(2));
  dPI(2) += -(f * mue) / (2. * pow(prinv(2), beta + 1.)) -
            (mue * (pow(prinv(2), beta - 1.) - prinv(1) / prinv(2) / prinv(2)) * (f - 1.)) * 0.5;

  ddPII(2) += (f * mue * (beta + 1.)) / (2. * pow(prinv(2), beta + 2.)) -
              (mue * (f - 1.) *
                  ((2. * prinv(1)) / prinv(2) / prinv(2) / prinv(2) +
                      pow(prinv(2), beta - 2.) * (beta - 1.))) *
                  0.5;
  ddPII(3) -= (1. - f) * 0.5 * mue / prinv(2) / prinv(2);
}

void Mat::Elastic::CoupBlatzKo::add_third_derivatives_principal_iso(
    Core::LinAlg::Matrix<10, 1>& dddPIII_iso, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  // material parameters for isochoric part
  const double mu = params_->mue_;   // Shear modulus
  const double nue = params_->nue_;  // Poisson's ratio
  const double f = params_->f_;      // interpolation parameter

  // introducing beta for consistency with Holzapfel and simplification
  const double beta = nue / (1. - 2. * nue);

  dddPIII_iso(2) +=
      -f * mu * pow(prinv(2), -beta - 3.) * beta * beta * .5 -
      1.5 * f * mu * pow(prinv(2), -beta - 3.) * beta - f * mu * pow(prinv(2), -beta - 3.) +
      (1. - f) * mu *
          (-6. * prinv(1) * pow(prinv(2), -4.) + pow(prinv(2), beta - 3.) * beta * beta -
              3. * pow(prinv(2), beta - 3.) * beta + 2. * pow(prinv(2), beta - 3.)) *
          .5;

  dddPIII_iso(8) += -(-1. + f) * mu * pow(prinv(2), -3.);
}

void Mat::Elastic::CoupBlatzKo::add_coup_deriv_vol(
    const double J, double* dPj1, double* dPj2, double* dPj3, double* dPj4)
{
  // material parameters for isochoric part
  const double mu = params_->mue_;   // Shear modulus
  const double nue = params_->nue_;  // Poisson's ratio
  const double f = params_->f_;      // interpolation parameter

  // introducing beta for consistency with Holzapfel and simplification
  const double beta = nue / (1. - 2. * nue);

  // generated with maple:
  if (dPj1)
    *dPj1 += f * mu * (2. * pow(J, -1. / 3.) - 2. * pow(J * J, -beta) / J) / 2. +
             (1. - f) * mu * (-2. * pow(J, -5. / 3.) + 2. * pow(J * J, beta) / J) / 2.;
  if (dPj2)
    *dPj2 += f * mu *
                 (-2. / 3. * pow(J, -4. / 3.) + 4. * pow(J * J, -beta) * beta * pow(J, -2.) +
                     2. * pow(J * J, -beta) * pow(J, -2.)) /
                 2. +
             (1. - f) * mu *
                 (10. / 3. * pow(J, -8. / 3.) + 4. * pow(J * J, beta) * beta * pow(J, -2.) -
                     2. * pow(J * J, beta) * pow(J, -2.)) /
                 2.;
  if (dPj3)
    *dPj3 +=
        f * mu *
            (8. / 9. * pow(J, -7. / 3.) - 8. * pow(J * J, -beta) * beta * beta * pow(J, -3.) -
                12. * pow(J * J, -beta) * beta * pow(J, -3.) -
                4. * pow(J * J, -beta) * pow(J, -3.)) /
            2. +
        (1. - f) * mu *
            (-80. / 9. * pow(J, -11. / 3.) + 8. * pow(J * J, beta) * beta * beta * pow(J, -3.) -
                12. * pow(J * J, beta) * beta * pow(J, -3.) + 4. * pow(J * J, beta) * pow(J, -3.)) /
            2.;
  if (dPj4)
    *dPj4 +=
        f * mu *
            (-56. / 27. * pow(J, -10. / 3.) +
                16. * pow(J * J, -beta) * pow(beta, 3.) * pow(J, -4.) +
                48. * pow(J * J, -beta) * beta * beta * pow(J, -4.) +
                44. * pow(J * J, -beta) * beta * pow(J, -4.) +
                12. * pow(J * J, -beta) * pow(J, -4.)) /
            2. +
        (1. - f) * mu *
            (880. / 27. * pow(J, -14. / 3.) + 16. * pow(J * J, beta) * pow(beta, 3.) * pow(J, -4.) -
                48. * pow(J * J, beta) * beta * beta * pow(J, -4.) +
                44. * pow(J * J, beta) * beta * pow(J, -4.) -
                12. * pow(J * J, beta) * pow(J, -4.)) /
            2.;
}
FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupmooneyrivlin.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupMooneyRivlin::CoupMooneyRivlin(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c1_(matdata.parameters.get<double>("C1")),
      c2_(matdata.parameters.get<double>("C2")),
      c3_(matdata.parameters.get<double>("C3"))
{
}

Mat::Elastic::CoupMooneyRivlin::CoupMooneyRivlin(Mat::Elastic::PAR::CoupMooneyRivlin* params)
    : params_(params)
{
}

void Mat::Elastic::CoupMooneyRivlin::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;
  const double c3 = params_->c3_;

  // strain energy: Psi = c_1 (I1 - 3)  +  c_2 (I2 - 3)  -  (2 c_1 + 4 c_2) ln(J) + c_3 * (J - 1)^2
  // add to overall strain energy

  psi += c1 * (prinv(0) - 3.) + c2 * (prinv(1) - 3.) - (2. * c1 + 4. * c2) * log(sqrt(prinv(2))) +
         c3 * pow((sqrt(prinv(2)) - 1.), 2.);
}

void Mat::Elastic::CoupMooneyRivlin::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;
  const double c3 = params_->c3_;

  dPI(0) += c1;
  dPI(1) += c2;
  dPI(2) += c3 * (1 - std::pow(prinv(2), -0.5)) - (c1 + 2. * c2) * std::pow(prinv(2), -1.);

  ddPII(2) += (c1 + 2 * c2) * std::pow(prinv(2), -2.) + 0.5 * c3 * std::pow(prinv(2), -1.5);
}

void Mat::Elastic::CoupMooneyRivlin::add_coup_deriv_vol(
    const double J, double* dPj1, double* dPj2, double* dPj3, double* dPj4)
{
  const double c1 = params_->c1_;
  const double c2 = params_->c2_;
  const double c3 = params_->c3_;

  // generated with maple:
  if (dPj1)
    *dPj1 += 2. * c1 * pow(J, -1. / 3.) + 4. * c2 * pow(J, 1. / 3.) - (2. * c1 + 4. * c2) / J +
             2. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -1. / 2.) * J;
  if (dPj2)
    *dPj2 += -2. / 3. * c1 * pow(J, -4. / 3.) + 4. / 3. * c2 * pow(J, -2. / 3.) +
             (2. * c1 + 4. * c2) * pow(J, -2.) + (2. * c3) -
             2. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -3. / 2.) * J * J +
             2. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -1. / 2.);
  if (dPj3)
    *dPj3 += 8. / 9. * c1 * pow(J, -0.7e1 / 3.) - 8. / 9. * c2 * pow(J, -5. / 3.) -
             2. * (2. * c1 + 4. * c2) * pow(J, -3.) +
             6. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -5. / 2.) * pow(J, 3.) -
             6. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -3. / 2.) * J;
  if (dPj4)
    *dPj4 += -56. / 27. * c1 * pow(J, -10. / 3.) + 40. / 27. * c2 * pow(J, -8. / 3.) +
             6. * (2. * c1 + 40 * c2) * pow(J, -4.) -
             30. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -7. / 2.) * pow(J, 4.) +
             36. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -5. / 2.) * J * J -
             6. * c3 * (sqrt(J * J) - 1.) * pow(J * J, -3. / 2.);
}
FOUR_C_NAMESPACE_CLOSE

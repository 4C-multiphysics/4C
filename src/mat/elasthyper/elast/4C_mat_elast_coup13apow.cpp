// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coup13apow.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::Coup13aPow::Coup13aPow(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      c_(matdata.parameters.get<double>("C")),
      d_(matdata.parameters.get<int>("D")),
      a_(matdata.parameters.get<double>("A"))
{
}

Mat::Elastic::Coup13aPow::Coup13aPow(Mat::Elastic::PAR::Coup13aPow* params) : params_(params) {}

void Mat::Elastic::Coup13aPow::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constants
  const double c = params_->c_;
  const int d = params_->d_;
  const double a = params_->a_;

  // strain energy: Psi = c (I_{\boldsymbol{C}}*(III_{\boldsymbol{C}}^(-a))-3)^d
  // add to overall strain energy
  psi += c * pow((prinv(0) * (pow(prinv(2), -a)) - 3.), d);
}

void Mat::Elastic::Coup13aPow::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double c = params_->c_;
  const int d = params_->d_;
  const double a = params_->a_;

  // If d<2 the material model is not stress free in the reference configuration
  if (d < 2)
    FOUR_C_THROW(
        "The Elast_Coup13aPow - material only works for positive integer exponents, which are "
        "larger than two.");
  // a==0 not necessary to implement extra as equal to coup1pow
  if (a == 0) FOUR_C_THROW("Use Elast_Coup1Pow.");
  // Material not implemented for negative exponents a
  if (a < 0) FOUR_C_THROW("Use positive values for the exponent a.");
  // Restriction for a (due to polyconvexity)
  if (a > (1. / 3. - 1. / (2. * d)))
    std::cout << "\nWARNING: A should be smaller then 1./3.- 1/(2*D). And D should be odd."
              << std::endl;

  const double I1I3a3 = prinv(0) * pow(prinv(2), -a) - 3.;

  dPI(0) += c * d * pow(prinv(2), -a) * pow(I1I3a3, d - 1.);
  dPI(2) += -a * c * d * prinv(0) * pow(prinv(2), -a - 1.) * pow(I1I3a3, d - 1.);

  if (d == 2)
  {
    ddPII(0) += c * d * (d - 1.) * pow(prinv(2), -2. * a);
    ddPII(2) += a * (a + 1.) * c * d * prinv(0) * pow(prinv(2), -a - 2.) * pow(I1I3a3, d - 1.) +
                a * a * c * d * (d - 1.) * prinv(0) * prinv(0) * pow(prinv(2), (-2. * a - 2.));
    ddPII(4) += -a * c * d * pow(prinv(2), -a - 1.) * pow(I1I3a3, d - 1.) -
                a * c * d * (d - 1.) * prinv(0) * pow(prinv(2), (-2. * a - 1.));
  }
  else
  {
    ddPII(0) += c * d * (d - 1.) * pow(prinv(2), -2. * a) * pow(I1I3a3, d - 2.);
    ddPII(2) += a * (a + 1.) * c * d * prinv(0) * pow(prinv(2), -a - 2.) * pow(I1I3a3, d - 1.) +
                a * a * c * d * (d - 1.) * prinv(0) * prinv(0) * pow(prinv(2), (-2. * a - 2.)) *
                    pow(I1I3a3, d - 2.);
    ddPII(4) +=
        -a * c * d * pow(prinv(2), -a - 1.) * pow(I1I3a3, d - 1.) -
        a * c * d * (d - 1.) * prinv(0) * pow(prinv(2), (-2. * a - 1.)) * pow(I1I3a3, d - 2.);
  }
}
FOUR_C_NAMESPACE_CLOSE

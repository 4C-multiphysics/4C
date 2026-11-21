// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_couplogmixneohooke.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupLogMixNeoHooke::CoupLogMixNeoHooke(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  std::string parmode = (matdata.parameters.get<std::string>("MODE"));
  double c1 = matdata.parameters.get<double>("C1");
  double c2 = matdata.parameters.get<double>("C2");

  if (parmode == "YN")
  {
    if (c2 <= 0.5 and c2 > -1.0)
    {
      lambda_ = (c2 == 0.5) ? 0.0 : c1 * c2 / ((1.0 + c2) * (1.0 - 2.0 * c2));
      mue_ = c1 / (2.0 * (1.0 + c2));  // shear modulus
    }
    else
      FOUR_C_THROW("Poisson's ratio must be between -1.0 and 0.5!");
  }
  else if (parmode == "Lame")
  {
    mue_ = c1;
    lambda_ = c2;
  }
  else
    FOUR_C_THROW(
        "unknown parameter set for NeoHooke material!\n Must be either YN (Young's modulus and "
        "Poisson's ratio) or Lame");
}

Mat::Elastic::CoupLogMixNeoHooke::CoupLogMixNeoHooke(Mat::Elastic::PAR::CoupLogMixNeoHooke* params)
    : params_(params)
{
}

void Mat::Elastic::CoupLogMixNeoHooke::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double lambda = params_->lambda_;
  const double mue = params_->mue_;
  const double sq = std::sqrt(prinv(2));

  // strain energy: Psi = \frac{\mu}{2} (I_{\boldsymbol{C}} - 3)
  //                      - \mu \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})
  //                      + \frac{\lambda}{2} \big((\sqrt{I\!I\!I_{\boldsymbol{C}}} - 1) \big)^2
  // add to overall strain energy
  psi += mue * 0.5 * (prinv(0) - 3.) - mue * log(sq) + lambda * 0.5 * pow((sq - 1.), 2.);
}

void Mat::Elastic::CoupLogMixNeoHooke::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double lambda = params_->lambda_;
  const double mue = params_->mue_;
  const double sq = std::sqrt(prinv(2));

  dPI(0) += mue * 0.5;
  dPI(2) += lambda * (sq - 1.) / (2. * sq) - mue / (2. * prinv(2));

  ddPII(2) += lambda / (4. * prinv(2)) + mue / (2. * prinv(2) * prinv(2)) -
              lambda * (sq - 1.) / (4. * sq * sq * sq);
}
FOUR_C_NAMESPACE_CLOSE

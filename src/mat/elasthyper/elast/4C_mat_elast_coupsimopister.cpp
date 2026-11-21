// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupsimopister.hpp"

#include "4C_material_parameter_base.hpp"

#include <limits>

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupSimoPister::CoupSimoPister(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), mue_(matdata.parameters.get<double>("MUE"))
{
}

Mat::Elastic::CoupSimoPister::CoupSimoPister(Mat::Elastic::PAR::CoupSimoPister* params)
    : params_(params)
{
}

void Mat::Elastic::CoupSimoPister::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  // material Constant mu
  const double mue = params_->mue_;

  // strain energy: \Psi = 0.5*\mu(I_1-3) - \mu log(J)
  // add to overall strain energy
  psi += 0.5 * mue * (prinv(0) - 3.) - mue * log(std::pow(prinv(2), 0.5));
}

void Mat::Elastic::CoupSimoPister::add_derivatives_principal(Core::LinAlg::Matrix<3, 1>& dPI,
    Core::LinAlg::Matrix<6, 1>& ddPII, const Core::LinAlg::Matrix<3, 1>& prinv, const int gp,
    const int eleGID)
{
  const double mue = params_->mue_;

  dPI(0) += 0.5 * mue;
  dPI(2) -= 0.5 * mue / prinv(2);

  ddPII(2) += 0.5 * mue / (prinv(2) * prinv(2));
}
FOUR_C_NAMESPACE_CLOSE

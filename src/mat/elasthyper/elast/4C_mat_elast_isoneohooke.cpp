// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isoneohooke.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::IsoNeoHooke::IsoNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata), mue_(matdata.parameters.get<Core::IO::InputField<double>>("MUE"))
{
}

Mat::Elastic::IsoNeoHooke::IsoNeoHooke(Mat::Elastic::PAR::IsoNeoHooke* params) : params_(params) {}

void Mat::Elastic::IsoNeoHooke::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double mue = params_->mue_.at(eleGID);

  // strain energy: Psi = frac{\mu}{2} (\overline{I}_{\boldsymbol{C}}-3) = \frac{\mu}{2}
  // (J^{-2/3}{I}_{\boldsymbol{C}}-3) add to overall strain energy
  psi += mue * 0.5 * (modinv(0) - 3.);
}


void Mat::Elastic::IsoNeoHooke::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double mue = params_->mue_.at(eleGID);

  dPmodI(0) += 0.5 * mue;
}

FOUR_C_NAMESPACE_CLOSE

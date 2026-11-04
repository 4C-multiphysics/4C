// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_volpenalty.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::VolPenalty::VolPenalty(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      eps_(matdata.parameters.get<double>("EPSILON")),
      game_(matdata.parameters.get<double>("GAMMA"))
{
  if (eps_ < 0. || game_ <= 0.)
    FOUR_C_THROW("VolPenalty parameters EPSILON and GAMMA have to be greater zero");
}

Mat::Elastic::VolPenalty::VolPenalty(Mat::Elastic::PAR::VolPenalty* params) : params_(params) {}

void Mat::Elastic::VolPenalty::add_strain_energy(double& psi,
    const Core::LinAlg::Matrix<3, 1>& prinv, const Core::LinAlg::Matrix<3, 1>& modinv,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp, const int eleGID)
{
  const double eps = params_->eps_;
  const double game = params_->game_;

  // strain energy: Psi=\epsilon \left( J^{\gamma} + \frac 1 {J^{\gamma}} -2 \right)
  // add to overall strain energy
  psi += eps * (pow(modinv(2), game) + pow(modinv(2), -game) - 2.);
}

void Mat::Elastic::VolPenalty::add_derivatives_modified(Core::LinAlg::Matrix<3, 1>& dPmodI,
    Core::LinAlg::Matrix<6, 1>& ddPmodII, const Core::LinAlg::Matrix<3, 1>& modinv, const int gp,
    const int eleGID)
{
  const double eps = params_->eps_;
  const double game = params_->game_;

  dPmodI(2) += eps * game * (pow(modinv(2), game - 1.) - pow(modinv(2), -game - 1.));

  ddPmodII(2) +=
      eps * game *
      ((game - 1.) * pow(modinv(2), game - 2.) + (game + 1.) * pow(modinv(2), -game - 2.));
}

void Mat::Elastic::VolPenalty::add3rd_vol_deriv(
    const Core::LinAlg::Matrix<3, 1>& modinv, double& d3PsiVolDJ3)
{
  const double eps = params_->eps_;
  const double game = params_->game_;
  const double J = modinv(2);
  d3PsiVolDJ3 += eps * (game * (game - 1.) * (game - 2.) * pow(J, game - 3.) +
                           (-game) * (-game - 1.) * (-game - 2.) * pow(J, -game - 3.));
}
FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_isovarga.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::IsoVarga::IsoVarga(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      mue_(matdata.parameters.get<double>("MUE")),
      beta_(matdata.parameters.get<double>("BETA"))
{
}

Mat::Elastic::IsoVarga::IsoVarga(Mat::Elastic::PAR::IsoVarga* params) : params_(params) {}

void Mat::Elastic::IsoVarga::add_coefficients_stretches_modified(
    Core::LinAlg::Matrix<3, 1>& modgamma, Core::LinAlg::Matrix<6, 1>& moddelta,
    const Core::LinAlg::Matrix<3, 1>& modstr)
{
  // parameters
  const double alpha = 2.0 * params_->mue_ - params_->beta_;
  const double beta = params_->beta_;

  // first derivatives
  // \frac{\partial Psi}{\partial \bar{\lambda}_1}
  modgamma(0) += alpha - beta / (modstr(0) * modstr(0));
  // \frac{\partial Psi}{\partial \bar{\lambda}_2}
  modgamma(1) += alpha - beta / (modstr(1) * modstr(1));
  // \frac{\partial Psi}{\partial \bar{\lambda}_3}
  modgamma(2) += alpha - beta / (modstr(2) * modstr(2));

  // second derivatives
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_1^2}
  moddelta(0) += 2.0 * beta / (modstr(0) * modstr(0) * modstr(0));
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_2^2}
  moddelta(1) += 2.0 * beta / (modstr(1) * modstr(1) * modstr(1));
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_3^2}
  moddelta(2) += 2.0 * beta / (modstr(2) * modstr(2) * modstr(2));
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_1 \partial\bar{\lambda}_2}
  moddelta(3) += 0.0;
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_2 \partial\bar{\lambda}_3}
  moddelta(4) += 0.0;
  // \frac{\partial^2 Psi}{\partial\bar{\lambda}_3 \partial\bar{\lambda}_1}
  moddelta(5) += 0.0;
}

FOUR_C_NAMESPACE_CLOSE

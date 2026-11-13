// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_coupvarga.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::Elastic::PAR::CoupVarga::CoupVarga(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      mue_(matdata.parameters.get<double>("MUE")),
      beta_(matdata.parameters.get<double>("BETA"))
{
}

Mat::Elastic::CoupVarga::CoupVarga(Mat::Elastic::PAR::CoupVarga* params) : params_(params) {}

void Mat::Elastic::CoupVarga::add_coefficients_stretches_principal(
    Core::LinAlg::Matrix<3, 1>& gamma,  ///< see above, [gamma_1, gamma_2, gamma_3]
    Core::LinAlg::Matrix<6, 1>&
        delta,  ///< see above, [delta_11, delta_22, delta_33, delta_12, delta_23, delta_31]
    const Core::LinAlg::Matrix<3, 1>&
        prstr  ///< principal stretches, [lambda_1, lambda_2, lambda_3]
)
{
  // parameters
  const double alpha = 2.0 * params_->mue_ - params_->beta_;
  const double beta = params_->beta_;

  // first derivatives
  // \frac{\partial Psi}{\partial \lambda_1}
  gamma(0) += alpha - beta / (prstr(0) * prstr(0));
  // \frac{\partial Psi}{\partial \lambda_2}
  gamma(1) += alpha - beta / (prstr(1) * prstr(1));
  // \frac{\partial Psi}{\partial \lambda_3}
  gamma(2) += alpha - beta / (prstr(2) * prstr(2));

  // second derivatives
  // \frac{\partial^2 Psi}{\partial\lambda_1^2}
  delta(0) += 2.0 * beta / (prstr(0) * prstr(0) * prstr(0));
  // \frac{\partial^2 Psi}{\partial\lambda_2^2}
  delta(1) += 2.0 * beta / (prstr(1) * prstr(1) * prstr(1));
  // \frac{\partial^2 Psi}{\partial\lambda_3^2}
  delta(2) += 2.0 * beta / (prstr(2) * prstr(2) * prstr(2));
  // \frac{\partial^2 Psi}{\partial\lambda_1 \partial\lambda_2}
  delta(3) += 0.0;
  // \frac{\partial^2 Psi}{\partial\lambda_2 \partial\lambda_3}
  delta(4) += 0.0;
  // \frac{\partial^2 Psi}{\partial\lambda_3 \partial\lambda_1}
  delta(5) += 0.0;
}
FOUR_C_NAMESPACE_CLOSE

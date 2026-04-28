// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_elast_visco_genmax.hpp"

#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

Mat::Elastic::PAR::GenMax::GenMax(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata),
      tau_(matdata.parameters.get<double>("TAU")),
      beta_(matdata.parameters.get<double>("BETA")),
      solve_(matdata.parameters.get<std::string>("SOLVE"))
{
  if (solve_ != "OneStepTheta" && solve_ != "ExponentialTimeDiscretization")
    FOUR_C_THROW(
        "Invalid input for SOLVE='{}' in VISCO_GenMax (MAT {}). Use OneStepTheta or "
        "ExponentialTimeDiscretization.",
        solve_, matdata.id);
}

Mat::Elastic::GenMax::GenMax(Mat::Elastic::PAR::GenMax* params) : params_(params) {}

void Mat::Elastic::GenMax::read_material_parameters_visco(
    double& tau, double& beta, double& alpha, std::string& solve)
{
  tau = params_->tau_;
  beta = params_->beta_;
  solve = params_->solve_;
}

FOUR_C_NAMESPACE_CLOSE

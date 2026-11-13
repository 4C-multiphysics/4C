// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_so3_material.hpp"

#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


double Mat::So3Material::strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
    const EvaluationContext<3>& context, int gp, int eleGID) const
{
  FOUR_C_THROW(
      "Material of type {} does not support calculation of strain energy", this->material_type());
}


double Mat::So3Material::evaluate_cauchy_n_dir_and_derivatives(
    const Core::LinAlg::Tensor<double, 3, 3>& defgrd, const Core::LinAlg::Tensor<double, 3>& n,
    const Core::LinAlg::Tensor<double, 3>& dir, Core::LinAlg::Matrix<3, 1>* d_cauchyndir_dn,
    Core::LinAlg::Matrix<3, 1>* d_cauchyndir_ddir, Core::LinAlg::Matrix<9, 1>* d_cauchyndir_dF,
    Core::LinAlg::Matrix<9, 9>* d2_cauchyndir_dF2, Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_dn,
    Core::LinAlg::Matrix<9, 3>* d2_cauchyndir_dF_ddir, const EvaluationContext<3>& context,
    int eleGID, const double* concentration)
{
  FOUR_C_THROW("evaluate_cauchy_n_dir_and_derivatives not implemented for material of type {}",
      this->material_type());
}


void Mat::So3Material::evaluate_linearization_od(const Core::LinAlg::Tensor<double, 3, 3>& defgrd,
    double concentration, Core::LinAlg::Matrix<9, 1>& d_F_dx)
{
  FOUR_C_THROW(
      "evaluate_linearization_od not implemented for material of type {}", this->material_type());
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mat_par_aniso.hpp"

#include "4C_comm_pack_helpers.hpp"
#include "4C_global_data.hpp"
#include "4C_mat_elast_aniso_structuraltensor_strategy.hpp"
#include "4C_mat_par_bundle.hpp"
#include "4C_utils_enum.hpp"

FOUR_C_NAMESPACE_OPEN


Mat::PAR::ParameterAniso::ParameterAniso(const Core::Mat::PAR::Parameter::Data& matdata)
    : Parameter(matdata)
{
  // get MAT ID for definition of structural tensor
  int mat_id_structural_tensor = matdata.parameters.get<int>("STR_TENS_ID");
  // get pointer to material
  auto* mat_str_tens =
      Global::Problem::instance()->materials()->parameter_by_id(mat_id_structural_tensor);
  // construct parameter class
  auto* params = static_cast<Mat::Elastic::PAR::StructuralTensorParameter*>(mat_str_tens);
  // get type of strategy
  std::string strategy = mat_str_tens->raw_parameters().get<std::string>("STRATEGY");

  // construct strategy
  if (strategy == "Standard")
  {
    structural_tensor_strategy_ =
        std::make_shared<Mat::Elastic::StructuralTensorStrategyStandard>(params);
  }
  else if (strategy == "ByDistributionFunction")
  {
    structural_tensor_strategy_ =
        std::make_shared<Mat::Elastic::StructuralTensorStrategyByDistributionFunction>(params);
  }
  else if (strategy == "DispersedTransverselyIsotropic")
  {
    structural_tensor_strategy_ =
        std::make_shared<Mat::Elastic::StructuralTensorStrategyDispersedTransverselyIsotropic>(
            params);
  }
  else
    FOUR_C_THROW("Unknown type of structural tensor strategy for anisotropic material chosen.");
}

FOUR_C_NAMESPACE_CLOSE

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_structure_new_predict_factory.hpp"

#include "4C_structure_new_input.hpp"
#include "4C_utils_exceptions.hpp"

// supported predictor classes
#include "4C_structure_new_predict_constdisvelaccpress.hpp"
#include "4C_structure_new_predict_python_wrapper.hpp"
#include "4C_structure_new_predict_tangdis.hpp"

FOUR_C_NAMESPACE_OPEN

/*----------------------------------------------------------------------------*
 *----------------------------------------------------------------------------*/
std::shared_ptr<Solid::Predict::Generic> Solid::Predict::build_predictor(
    const Solid::PredictorType& predType)
{
  std::shared_ptr<Solid::Predict::Generic> predictor = nullptr;

  switch (predType)
  {
    case Solid::PredictorType::constdis:
    case Solid::PredictorType::constvel:
    case Solid::PredictorType::constacc:
    case Solid::PredictorType::constdisvelacc:
    case Solid::PredictorType::constdispres:
    case Solid::PredictorType::constdisvelaccpres:
      predictor = std::make_shared<Solid::Predict::ConstDisVelAccPress>();
      break;
    case Solid::PredictorType::tangdis:
    case Solid::PredictorType::tangdis_constfext:
      predictor = std::make_shared<Solid::Predict::TangDis>();
      break;
    case Solid::PredictorType::python_wrapper:
#ifdef FOUR_C_WITH_PYBIND11
      predictor = std::make_shared<Solid::Predict::PythonWrapper>();
#else
      FOUR_C_THROW(
          "The 'PythonWrapper' predictor type requires 4C to be compiled with pybind11 support, "
          "but pybind11 was not found during the configuration of 4C. Please either reconfigure 4C "
          "with pybind11 support or choose a different predictor type.");
#endif
      break;
    default:
      FOUR_C_THROW("Unknown predictor type!");
      break;
  }

  return predictor;
}

FOUR_C_NAMESPACE_CLOSE

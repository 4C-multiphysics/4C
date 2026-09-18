// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_INELASTIC_DEFGRAD_FACTORS_TEST_UTILS_HPP
#define FOUR_C_INELASTIC_DEFGRAD_FACTORS_TEST_UTILS_HPP

#include "4C_mat_inelastic_defgrad_factors_service.hpp"

namespace FourC::InelasticDefgradFactorsTestUtils
{
  namespace AEI =
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolation;

  /// default parameters for Adaptive Estimate Interpolation to be used in the tests
  struct DefaultAEIParams
  {
    bool use_adaptive_estimate_interpolation = true;
    double elastic_predictor_zero_component_threshold = 1.0e-13;
    AEI::PlasticPredictorConstructionParams plastic_predictor_construction{
        .elastic_stretch_eigenval_type =
            AEI::PrelimPlasticPredictor::ElasticStretchEigenvalType::scale_unit,
        .elastic_stretch_eigenvect_type =
            AEI::PrelimPlasticPredictor::ElasticStretchEigenvectType::from_elastic_predictor,
        .elastic_rotation_type =
            AEI::PrelimPlasticPredictor::ElasticRotationType::from_elastic_predictor,
        .max_iter = 50,
        .relative_understress_tol = 1.0e-6,
        .interval_scanning_param = 0.5,
    };
    AEI::EstimateInterpolationParams estimate_interpolation{
        .starting_point_type = AEI::StartingPointType::equiv_stress_history,
        .user_set_starting_point = 0.5,
        .max_iter = 50,
        .interval_scanning_param = 0.5,
    };
    AEI::HardeningParams hardening{
        .method = AEI::HardeningManagementMethod::integrate_via_evolution_equations,
        .max_iter_integration = 50,
        .tol_integration = 1.0e-8,
    };
    AEI::ReestimationParams reestimation{
        .max_num_reestimations = 10,
        .interval_scanning_param = 0.5,
    };
  };

  /// setup Adaptive Estimate Interpolation parameters using a config object
  inline AEI::AEIParams set_up_aei_params(const DefaultAEIParams& default_params = {})
  {
    return AEI::AEIParams{
        .use_adaptive_estimate_interpolation = default_params.use_adaptive_estimate_interpolation,
        .elastic_predictor_zero_component_threshold =
            default_params.elastic_predictor_zero_component_threshold,
        .plastic_predictor_construction = default_params.plastic_predictor_construction,
        .estimate_interpolation = default_params.estimate_interpolation,
        .hardening = default_params.hardening,
        .reestimation = default_params.reestimation,
    };
  }

}  // namespace FourC::InelasticDefgradFactorsTestUtils
#endif

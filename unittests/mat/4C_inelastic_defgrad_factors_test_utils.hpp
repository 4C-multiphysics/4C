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
  namespace AEINamespace =
      Mat::InelasticDefgradTransvIsotropElastViscoplastUtils::AdaptiveEstimateInterpolation;

  /// default config for Adaptive Estimate Interpolation parameters
  struct AEIParamsConfig
  {
    bool use_adaptive_estimate_interpolation = true;
    AEINamespace::PreconditioningSettings preconditioning{
        .precondition_elastic_pred = true,
        .tol_precondition_elastic_pred = 1.0e-13,
    };
    AEINamespace::PlasticPredictorConstructionParams plastic_predictor_construction{
        .elastic_stretch_eigenval_type =
            AEINamespace::PrelimPlasticPredictor::ElasticStretchEigenvalType::scale_unit,
        .elastic_stretch_eigenvect_type = AEINamespace::PrelimPlasticPredictor::
            ElasticStretchEigenvectType::from_elastic_predictor,
        .elastic_rotation_type =
            AEINamespace::PrelimPlasticPredictor::ElasticRotationType::from_elastic_predictor,
        .max_iter = 50,
        .relative_understress_tol = 1.0e-6,
        .interval_scanning_param = 0.5,
    };
    AEINamespace::EstimateInterpolationParams estimate_interpolation{
        .starting_point_type = AEINamespace::StartingPointType::equiv_stress_history,
        .user_set_starting_point = 0.5,
        .max_iter = 50,
        .interval_scanning_param = 0.5,
    };
    AEINamespace::HardeningParams hardening{
        .method = AEINamespace::HardeningMethod::integrate_via_evol_eqs,
        .max_iter_integration = 50,
        .tol_integration = 1.0e-8,
    };
    AEINamespace::ReestimationParams reestimation{
        .max_num_reestimations = 10,
        .interval_scanning_param = 0.5,
    };
  };

  /// setup Adaptive Estimate Interpolation parameters using a config object
  inline AEINamespace::AEIParams set_up_aei_params(const AEIParamsConfig& cfg = {})
  {
    return AEINamespace::AEIParams{
        .use_adaptive_estimate_interpolation = cfg.use_adaptive_estimate_interpolation,
        .preconditioning = cfg.preconditioning,
        .plastic_predictor_construction = cfg.plastic_predictor_construction,
        .estimate_interpolation = cfg.estimate_interpolation,
        .hardening = cfg.hardening,
        .reestimation = cfg.reestimation,
    };
  }

}  // namespace FourC::InelasticDefgradFactorsTestUtils
#endif

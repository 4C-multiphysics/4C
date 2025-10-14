// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_growth_strategy_isotropic.hpp"

#include "4C_linalg_tensor_generators.hpp"
#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::IsotropicGrowthStrategy::IsotropicGrowthStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Mixture::PAR::MixtureGrowthStrategy(matdata)
{
}

std::unique_ptr<Mixture::MixtureGrowthStrategy>
Mixture::PAR::IsotropicGrowthStrategy::create_growth_strategy()
{
  return std::make_unique<Mixture::IsotropicGrowthStrategy>();
}

void Mixture::IsotropicGrowthStrategy::evaluate_inverse_growth_deformation_gradient(
    Core::LinAlg::Tensor<double, 3, 3>& iFgM, const Mixture::MixtureRule& mixtureRule,
    double currentReferenceGrowthScalar, const Mat::EvaluationContext& context, int gp,
    int eleGID) const
{
  iFgM = std::pow(currentReferenceGrowthScalar, -1.0 / 3.0) *
         Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

void Mixture::IsotropicGrowthStrategy::evaluate_growth_stress_cmat(
    const Mixture::MixtureRule& mixtureRule, double currentReferenceGrowthScalar,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& dCurrentReferenceGrowthScalarDC,
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID) const
{
  S_stress = {};
  cmat = {};
}
FOUR_C_NAMESPACE_CLOSE

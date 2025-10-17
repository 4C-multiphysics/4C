// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#include "4C_mixture_growth_strategy_stiffness.hpp"

#include "4C_linalg_fixedsizematrix_generators.hpp"
#include "4C_linalg_fixedsizematrix_tensor_products.hpp"
#include "4C_linalg_fixedsizematrix_voigt_notation.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

Mixture::PAR::StiffnessGrowthStrategy::StiffnessGrowthStrategy(
    const Core::Mat::PAR::Parameter::Data& matdata)
    : Mixture::PAR::MixtureGrowthStrategy(matdata), kappa_(matdata.parameters.get<double>("KAPPA"))
{
}

std::unique_ptr<Mixture::MixtureGrowthStrategy>
Mixture::PAR::StiffnessGrowthStrategy::create_growth_strategy()
{
  return std::make_unique<Mixture::StiffnessGrowthStrategy>(this);
}

Mixture::StiffnessGrowthStrategy::StiffnessGrowthStrategy(
    Mixture::PAR::StiffnessGrowthStrategy* params)
    : params_(params)
{
}

void Mixture::StiffnessGrowthStrategy::evaluate_inverse_growth_deformation_gradient(
    Core::LinAlg::Tensor<double, 3, 3>& iFgM, const Mixture::MixtureRule& mixtureRule,
    double currentReferenceGrowthScalar, const Mat::EvaluationContext& context, int gp,
    int eleGID) const
{
  iFgM = Core::LinAlg::get_full(Core::LinAlg::TensorGenerators::identity<double, 3, 3>);
}

void Mixture::StiffnessGrowthStrategy::evaluate_growth_stress_cmat(
    const Mixture::MixtureRule& mixtureRule, double currentReferenceGrowthScalar,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& dCurrentReferenceGrowthScalarDC,
    const Core::LinAlg::Tensor<double, 3, 3>& F,
    const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
    const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
    Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp, const int eleGID) const
{
  Core::LinAlg::SymmetricTensor<double, 3, 3> iC =
      Core::LinAlg::assume_symmetry(Core::LinAlg::inv(Core::LinAlg::transpose(F) * F));

  const double kappa = params_->kappa_;
  const double detF = Core::LinAlg::det(F);
  const double I3 = detF * detF;

  const double dPi = 0.5 * kappa * (1.0 - currentReferenceGrowthScalar / detF);
  const double ddPi = 0.25 * kappa * currentReferenceGrowthScalar / std::pow(detF, 3);

  const double ddPiDGrowthScalar = -0.5 * kappa / detF;

  const double gamma2 = 2.0 * I3 * dPi;
  const double dgamma2DGrowthScalar = 4.0 * I3 * ddPiDGrowthScalar;
  const double delta6 = 4. * (I3 * dPi + I3 * I3 * ddPi);
  const double delta7 = -4.0 * I3 * dPi;


  S_stress = gamma2 * iC;

  // contribution: Cinv \otimes Cinv
  cmat = delta6 * Core::LinAlg::dyadic(iC, iC);
  // contribution: Cinv \odot Cinvs
  cmat += delta7 * Core::LinAlg::FourTensorOperations::holzapfel_product(iC);

  cmat += dgamma2DGrowthScalar * Core::LinAlg::dyadic(iC, dCurrentReferenceGrowthScalarDC);
}
FOUR_C_NAMESPACE_CLOSE

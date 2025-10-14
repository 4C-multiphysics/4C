// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_GROWTH_STRATEGY_ISOTROPIC_HPP
#define FOUR_C_MIXTURE_GROWTH_STRATEGY_ISOTROPIC_HPP

#include "4C_config.hpp"

#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class IsotropicGrowthStrategy;

  namespace PAR
  {
    class IsotropicGrowthStrategy : public Mixture::PAR::MixtureGrowthStrategy
    {
     public:
      explicit IsotropicGrowthStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      std::unique_ptr<Mixture::MixtureGrowthStrategy> create_growth_strategy() override;
    };
  }  // namespace PAR

  /*!
   * @brief Growth is modeled as an inelastic volumentric expansion of the whole cell (isotropic).
   */
  class IsotropicGrowthStrategy : public Mixture::MixtureGrowthStrategy
  {
   public:
    [[nodiscard]] bool has_inelastic_growth_deformation_gradient() const override { return true; };

    void evaluate_inverse_growth_deformation_gradient(Core::LinAlg::Tensor<double, 3, 3>& iFgM,
        const Mixture::MixtureRule& mixtureRule, double currentReferenceGrowthScalar,
        const Mat::EvaluationContext& context, int gp, int eleGID) const override;

    void evaluate_growth_stress_cmat(const Mixture::MixtureRule& mixtureRule,
        double currentReferenceGrowthScalar,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& dCurrentReferenceGrowthScalarDC,
        const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp,
        const int eleGID) const override;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_GROWTH_STRATEGY_STIFFNESS_HPP
#define FOUR_C_MIXTURE_GROWTH_STRATEGY_STIFFNESS_HPP

#include "4C_config.hpp"

#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class StiffnessGrowthStrategy;

  namespace PAR
  {
    class StiffnessGrowthStrategy : public Mixture::PAR::MixtureGrowthStrategy
    {
     public:
      explicit StiffnessGrowthStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      std::unique_ptr<Mixture::MixtureGrowthStrategy> create_growth_strategy() override;

      const double kappa_;
    };
  }  // namespace PAR

  /*!
   * @brief Growth modeled as an elastic expansion of the whole cell with a penalty type formulation
   *
   * This growth strategy uses a penalty term that ensures constant spatial density. The penalty
   * term has the following form:
   *
   * \f[
   *     \Psi = \frac{1}{2} \kappa (|F| - \frac{\rho_0(s)}{\rho_0(s=0)})^2
   * \f]
   *
   * The model is based on Braeu et al (2019),
   * https://link.springer.com/article/10.1007%2Fs10237-018-1084-x
   */
  class StiffnessGrowthStrategy : public Mixture::MixtureGrowthStrategy
  {
   public:
    explicit StiffnessGrowthStrategy(Mixture::PAR::StiffnessGrowthStrategy* params);

    [[nodiscard]] bool has_inelastic_growth_deformation_gradient() const override { return false; };

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

   private:
    ///! growth parameters as defined in the input file
    const PAR::StiffnessGrowthStrategy* params_{};
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
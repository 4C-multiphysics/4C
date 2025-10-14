// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_GROWTH_STRATEGY_ANISOTROPIC_HPP
#define FOUR_C_MIXTURE_GROWTH_STRATEGY_ANISOTROPIC_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mixture_growth_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class AnisotropicGrowthStrategy;

  namespace PAR
  {
    class AnisotropicGrowthStrategy : public Mixture::PAR::MixtureGrowthStrategy
    {
     public:
      explicit AnisotropicGrowthStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      std::unique_ptr<Mixture::MixtureGrowthStrategy> create_growth_strategy() override;

      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          growth_direction;

      /// structural tensor strategy
      std::shared_ptr<Mat::Elastic::StructuralTensorStrategyBase> structural_tensor_strategy_;
    };
  }  // namespace PAR

  /*!
   * @brief Growth is modeled as an inelastic expansion of the whole cell in one predefined
   * direction
   *
   * The direction of growth can be specified with a fiber with the fiber id specified in FIBER_ID
   *
   */
  class AnisotropicGrowthStrategy : public Mixture::MixtureGrowthStrategy
  {
   public:
    explicit AnisotropicGrowthStrategy(Mixture::PAR::AnisotropicGrowthStrategy* params);

    void pack_mixture_growth_strategy(Core::Communication::PackBuffer& data) const override;

    void unpack_mixture_growth_strategy(Core::Communication::UnpackBuffer& buffer) override;

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

   private:
    ///! growth parameters as defined in the input file
    const PAR::AnisotropicGrowthStrategy* params_{};

    mutable std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> structural_tensors_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
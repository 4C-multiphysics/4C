// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_GROWTH_STRATEGY_HPP
#define FOUR_C_MIXTURE_GROWTH_STRATEGY_HPP

#include "4C_config.hpp"

#include "4C_mat_so3_material.hpp"
#include "4C_mixture_rule.hpp"

#include <memory>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class MixtureGrowthStrategy;
  namespace PAR
  {
    class MixtureGrowthStrategy : public Core::Mat::PAR::Parameter
    {
      friend class Mixture::MixtureGrowthStrategy;

     public:
      /// constructor
      explicit MixtureGrowthStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      /// Override this method and throw error, as only the create_growth_strategy() should be used.
      std::shared_ptr<Core::Mat::Material> create_material() final
      {
        FOUR_C_THROW(
            "Cannot create mixture growth strategy from this method. Use create_growth_strategy() "
            "instead.");
        return nullptr;
      }

      /// create material instance of matching type with my parameters
      virtual std::unique_ptr<Mixture::MixtureGrowthStrategy> create_growth_strategy() = 0;

      /*!
       * \brief Factory of the mixture growth strategy parameters
       *
       * This static method generates the specific class of the mixture growth strategy defined in
       * the datfile at the corresponding material id
       *
       * @param matid Material id of the mixturerule
       * @return Parameters of the referenced mixture rule
       */
      static Mixture::PAR::MixtureGrowthStrategy* factory(int matid);
    };
  }  // namespace PAR

  class MixtureGrowthStrategy
  {
   public:
    virtual ~MixtureGrowthStrategy() = default;

    MixtureGrowthStrategy() = default;
    MixtureGrowthStrategy(const MixtureGrowthStrategy& copyFrom) = default;
    MixtureGrowthStrategy& operator=(const MixtureGrowthStrategy& copyFrom) = default;
    MixtureGrowthStrategy(MixtureGrowthStrategy&&) noexcept = default;
    MixtureGrowthStrategy& operator=(MixtureGrowthStrategy&&) noexcept = default;

    virtual void pack_mixture_growth_strategy(Core::Communication::PackBuffer& data) const {}

    virtual void unpack_mixture_growth_strategy(Core::Communication::UnpackBuffer& buffer) {}

    virtual void register_anisotropy_extensions(Mat::Anisotropy& anisotropy)
    {
      // do nothing in the default case
    }

    [[nodiscard]] virtual bool has_inelastic_growth_deformation_gradient() const = 0;

    /*!
     * @brief Evaluates the inverse growth deformation gradient at the Gausspoint #gp
     *
     * The growth deformation gradient describes the deformation of the solid by addition/removal
     * of materials.
     *
     * @param iFgM (out) : Inverse of the growth deformation gradient
     * @param mixtureRule (in) : mixture rule
     * @param currentReferenceGrowthScalar (in) : current reference growth scalar
     * @param context (in) : Container for additional information
     * @param gp (in) : Gauss point
     * @param eleGID (in) : global element id
     */
    virtual void evaluate_inverse_growth_deformation_gradient(
        Core::LinAlg::Tensor<double, 3, 3>& iFgM, const Mixture::MixtureRule& mixtureRule,
        double currentReferenceGrowthScalar, const Mat::EvaluationContext& context, int gp,
        int eleGID) const = 0;

    /*!
     * @brief Evaluates the contribution of the growth strategy to the stress tensor and the
     * linearization.
     *
     * This is meant for growth strategies that use some kind of penalty formulation to ensure
     * growth
     *
     * @param mixtureRule (in) : mixture rule
     * @param currentReferenceGrowthScalar (in) : current reference growth scalar (volume change in
     * percent)
     * @param dCurrentReferenceGrowthScalarDC (in) : Derivative of the current reference growth
     * scalar w.r.t. Cauchy green deformation tensor
     * @param F (in) : deformation gradient
     * @param E_strain (in) : Green-Lagrange strain tensor
     * @param params (in) : Container for additional information
     * @param S_stress (out) : 2nd Piola-Kirchhoff stress tensor in stress like Voigt notation
     * @param cmat (out) : linearization of the 2nd Piola-Kirchhoff stress tensor
     * @param gp (in) : Gauss point
     * @param eleGID (in) : global element id
     */
    virtual void evaluate_growth_stress_cmat(const Mixture::MixtureRule& mixtureRule,
        double currentReferenceGrowthScalar,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& dCurrentReferenceGrowthScalarDC,
        const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E_strain,
        const Teuchos::ParameterList& params, Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, const int gp,
        const int eleGID) const = 0;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
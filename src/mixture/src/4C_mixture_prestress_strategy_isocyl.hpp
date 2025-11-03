// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ISOCYL_HPP
#define FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ISOCYL_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_mixture_prestress_strategy.hpp"

#include <NOX_LAPACK.H>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  struct EvaluationContext;
}
namespace Mixture
{
  // forward declaration
  class IsotropicCylinderPrestressStrategy;

  namespace PAR
  {
    class IsotropicCylinderPrestressStrategy : public Mixture::PAR::PrestressStrategy
    {
      friend class Mixture::IsotropicCylinderPrestressStrategy;

     public:
      /// constructor
      explicit IsotropicCylinderPrestressStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create prestress strategy instance of matching type with my parameters
      std::unique_ptr<Mixture::PrestressStrategy> create_prestress_strategy() override;

      /// @name parameters of the prestress strategy
      /// @{
      const double inner_radius_;
      const double wall_thickness_;
      const double axial_prestretch_;
      const double circumferential_prestretch_;
      const double pressure_;

      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          radial;
      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          axial;
      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          circumferential;
      /// @}
    };
  }  // namespace PAR


  /*!
   * \brief Prestressing strategy for an isotropic constituent as part of the cylinder.
   *
   * \note This method also provides the possibility to setup equilibrium via membrane sub-parts
   */
  class IsotropicCylinderPrestressStrategy : public PrestressStrategy,
                                             public ElastinMembranePrestressStrategy
  {
   public:
    /// Constructor for the material given the material parameters
    explicit IsotropicCylinderPrestressStrategy(
        Mixture::PAR::IsotropicCylinderPrestressStrategy* params);
    void setup(Mixture::MixtureConstituent& constituent, const Teuchos::ParameterList& params,
        int gp, int eleGID) override;

    double evaluate_mue_frac(MixtureRule& mixtureRule,
        const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
        Mixture::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
        const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp,
        int eleGID) const override;

    void evaluate_prestress(const MixtureRule& mixtureRule,
        const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
        Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
        const Teuchos::ParameterList& params, const Mat::EvaluationContext& context, int gp,
        int eleGID) override;

    void update(const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
        Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext& context, int gp, int eleGID) override;

   private:
    /// Holder for internal parameters
    const PAR::IsotropicCylinderPrestressStrategy* params_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif

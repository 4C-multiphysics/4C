// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_PRESTRESS_STRATEGY_CONSTANT_HPP
#define FOUR_C_MIXTURE_PRESTRESS_STRATEGY_CONSTANT_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_mixture_prestress_strategy.hpp"

#include <NOX_LAPACK.H>

#include <array>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  // forward declaration
  class ConstantPrestressStrategy;

  namespace PAR
  {
    class ConstantPrestressStrategy : public Mixture::PAR::PrestressStrategy
    {
      friend class Mixture::ConstantPrestressStrategy;

     public:
      /// constructor
      explicit ConstantPrestressStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create prestress strategy instance of matching type with my parameters
      std::unique_ptr<Mixture::PrestressStrategy> create_prestress_strategy() override;

      /// @name parameters of the prestress strategy
      /// @{
      std::array<double, 6> prestretch_;
      /// @}
    };
  }  // namespace PAR


  /*!
   * \brief Prestressing strategy for a constant predefined prestretch tensor.
   */
  class ConstantPrestressStrategy : public PrestressStrategy
  {
   public:
    /// Constructor for the material given the material parameters
    explicit ConstantPrestressStrategy(Mixture::PAR::ConstantPrestressStrategy* params);

    void setup(Mixture::MixtureConstituent& constituent, const Teuchos::ParameterList& params,
        int gp, int eleGID) override;

    void evaluate_prestress(const MixtureRule& mixtureRule,
        const std::shared_ptr<const Mat::CoordinateSystemProvider> cosy,
        Mixture::MixtureConstituent& constituent, Core::LinAlg::SymmetricTensor<double, 3, 3>& G,
        const Teuchos::ParameterList& params, int gp, int eleGID) override;

    void update(const std::shared_ptr<const Mat::CoordinateSystemProvider> anisotropy,
        Mixture::MixtureConstituent& constituent, const Core::LinAlg::Tensor<double, 3, 3>& F,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& G, const Teuchos::ParameterList& params,
        int gp, int eleGID) override;

   private:
    /// Holder for internal parameters
    const PAR::ConstantPrestressStrategy* params_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif

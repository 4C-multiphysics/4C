// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ITERATIVE_HPP
#define FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ITERATIVE_HPP

#include "4C_config.hpp"

#include "4C_comm_pack_buffer.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_prestress_strategy.hpp"

FOUR_C_NAMESPACE_OPEN

// forward declarations
namespace Mat
{
  class CoordinateSystemProvider;
}

namespace MIXTURE
{
  // forward declaration
  class IterativePrestressStrategy;
  class MixtureConstituent;

  namespace PAR
  {
    class IterativePrestressStrategy : public MIXTURE::PAR::PrestressStrategy
    {
      friend class MIXTURE::IterativePrestressStrategy;

     public:
      /// constructor
      explicit IterativePrestressStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create prestress strategy instance of matching type with my parameters
      std::unique_ptr<MIXTURE::PrestressStrategy> create_prestress_strategy() override;


      /// @name parameters of the prestress strategy
      /// @{

      /// Flag whether the prestretch tensor is isochoric
      const bool isochoric_;
      /// Flag whether the prestretch tensor should be updated
      const bool is_active_;
      /// @}
    };
  }  // namespace PAR

  /*!
   * \brief Mixture prestress strategy to be used with MATERIAL_ITERATIVE prestressing
   *
   * This prestressing technique has to be used with the PRESTRESSTYPE MATERIAL_ITERATIVE. In each
   * prestress update step, the internal prestretch tensor is updated with the current stretch
   * tensor of the deformation.
   */
  class IterativePrestressStrategy : public PrestressStrategy
  {
   public:
    /// Constructor for the material given the material parameters
    explicit IterativePrestressStrategy(MIXTURE::PAR::IterativePrestressStrategy* params);

    void setup(MIXTURE::MixtureConstituent& constituent, Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    void evaluate_prestress(const MixtureRule& mixtureRule,
        const Teuchos::RCP<const Mat::CoordinateSystemProvider> anisotropy,
        MIXTURE::MixtureConstituent& constituent, Core::LinAlg::Matrix<3, 3>& G,
        Teuchos::ParameterList& params, int gp, int eleGID) override;

    void update(const Teuchos::RCP<const Mat::CoordinateSystemProvider> anisotropy,
        MIXTURE::MixtureConstituent& constituent, const Core::LinAlg::Matrix<3, 3>& F,
        Core::LinAlg::Matrix<3, 3>& G, Teuchos::ParameterList& params, int gp, int eleGID) override;

   private:
    /// Holder for internal parameters
    const PAR::IterativePrestressStrategy* params_;
  };
}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif

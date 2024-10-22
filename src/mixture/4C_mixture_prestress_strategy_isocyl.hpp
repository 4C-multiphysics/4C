// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ISOCYL_HPP
#define FOUR_C_MIXTURE_PRESTRESS_STRATEGY_ISOCYL_HPP

#include "4C_config.hpp"

#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_mixture_elastin_membrane_prestress_strategy.hpp"
#include "4C_mixture_prestress_strategy.hpp"

#include <NOX_LAPACK.H>

FOUR_C_NAMESPACE_OPEN

namespace MIXTURE
{
  // forward declaration
  class IsotropicCylinderPrestressStrategy;

  namespace PAR
  {
    class IsotropicCylinderPrestressStrategy : public MIXTURE::PAR::PrestressStrategy
    {
      friend class MIXTURE::IsotropicCylinderPrestressStrategy;

     public:
      /// constructor
      explicit IsotropicCylinderPrestressStrategy(const Core::Mat::PAR::Parameter::Data& matdata);

      /// create prestress strategy instance of matching type with my parameters
      std::unique_ptr<MIXTURE::PrestressStrategy> create_prestress_strategy() override;

      /// @name parameters of the prestress strategy
      /// @{
      const double inner_radius_;
      const double wall_thickness_;
      const double axial_prestretch_;
      const double circumferential_prestretch_;
      const double pressure_;
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
        MIXTURE::PAR::IsotropicCylinderPrestressStrategy* params);

    void setup(MIXTURE::MixtureConstituent& constituent, Teuchos::ParameterList& params, int gp,
        int eleGID) override;

    /*!
     * @brief Evaluates the prestretch
     * @param G (out) :  Prestretch of the constituent
     * @param params (in) : Container for additional information
     * @param gp (in) : Gauss-point
     * @param eleGID (in) : Global element id
     */
    void evaluate_prestress(const MixtureRule& mixtureRule,
        const Teuchos::RCP<const Mat::CoordinateSystemProvider> cosy,
        MIXTURE::MixtureConstituent& constituent, Core::LinAlg::Matrix<3, 3>& G,
        Teuchos::ParameterList& params, int gp, int eleGID) override;

    /*!
     * \brief Ensures equilibrium by adding a spacially variying part of the membrane
     *
     * \param mixtureRule Mixture rule
     * \param anisotropy Cylinder coordinate system
     * \param constituent Constituent that needs to be prestressed
     * \param membraneEvaluation Evaluator of the membrane sub-part
     * \param params Container for additional information
     * \param gp Gauss point
     * \param eleGID global Element id
     * \return double Fraction of the membrane stress contribution to ensure equilibrium
     */
    double evaluate_mue_frac(MixtureRule& mixtureRule,
        const Teuchos::RCP<const Mat::CoordinateSystemProvider> cosy,
        MIXTURE::MixtureConstituent& constituent, ElastinMembraneEvaluation& membraneEvaluation,
        Teuchos::ParameterList& params, int gp, int eleGID) const override;

    void update(const Teuchos::RCP<const Mat::CoordinateSystemProvider> anisotropy,
        MIXTURE::MixtureConstituent& constituent, const Core::LinAlg::Matrix<3, 3>& F,
        Core::LinAlg::Matrix<3, 3>& G, Teuchos::ParameterList& params, int gp, int eleGID) override;

   private:
    /// Holder for internal parameters
    const PAR::IsotropicCylinderPrestressStrategy* params_;
  };
}  // namespace MIXTURE

FOUR_C_NAMESPACE_CLOSE

#endif

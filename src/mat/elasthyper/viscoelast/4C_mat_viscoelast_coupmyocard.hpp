// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_VISCOELAST_COUPMYOCARD_HPP
#define FOUR_C_MAT_VISCOELAST_COUPMYOCARD_HPP

#include "4C_config.hpp"

#include "4C_mat_viscoelast_summand.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace ViscoElast
  {
    namespace PAR
    {
      /*!
       * @brief Parameters for the coupled myocardial matrix viscous summand.
       *
       * The parameter object stores the viscosity-like coefficient used by CoupMyocard and is
       * consumed through the visco summand factory.
       */
      class CoupMyocard : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupMyocard(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Viscosity-like coefficient of the myocardial matrix response.
        double n_;

        //@}

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        std::shared_ptr<Core::Mat::Material> create_material() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "Mat::ViscoElast::Summand::Factory.");
          return nullptr;
        };
      };  // class CoupMyocard
    }  // namespace PAR

    /*!
     * @brief Iso-rate style coupled viscous summand for myocardial matrix response.
     *
     * Within Mat::ViscoElastHyper, this summand activates IsoRateContribution and contributes
     * principal-rate coefficients to the common iso-rate evaluation path.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi_v = \eta/2 tr(\dot{E}^2) = \eta/8 tr(\dot{C}^2).
     * \f]
     *
     * Viscous second Piola-Kirchhoff stress
     * \f[
     *   S_v =  2 \frac{\partial \Psi_v}{\partial \dot{C}} = \eta/2 \dot{C}.
     * \f]
     *
     * Viscous constitutive tensor
     * \f[
     *   C_v =  4 \frac{\partial^2 W_v}{\partial \dot{C} \partial \dot{C}} = \eta I^\#,
     * \f]
     *
     * with
     *
     * \f[
     *   I^\#_{ijkl} = \frac{1}{2}(\delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk})
     * \f]
     */
    class CoupMyocard : public Mat::ViscoElast::Summand
    {
     public:
      /// constructor with given material parameters
      CoupMyocard(Mat::ViscoElast::PAR::CoupMyocard* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupmyocard;
      }

      //@}

      /// Add modified coeffiencts.
      void add_coefficients_visco_principal(
          const Core::LinAlg::Matrix<3, 1>& prinv,  ///< invariants of right Cauchy-Green tensor
          Core::LinAlg::Matrix<8, 1>& mu,   ///< necessary coefficients for piola-kirchhoff-stress
          Core::LinAlg::Matrix<33, 1>& xi,  ///< necessary coefficients for viscosity tensor
          Core::LinAlg::Matrix<7, 1>& rateinv, const Teuchos::ParameterList& params, double dt,
          int gp, int eleGID) override;

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
        isoprinc = true;
        viscogeneral = true;
        return;
      };

      /// Indicator for the chosen viscoelastic formulations
      void specify_visco_formulation(
          bool& visco_iso_rate,  ///< global indicator for isotropic rate-dependent visco response
          bool& visco_generalized_maxwell,  ///< global indicator for generalized Maxwell model
          bool& visco_quasi_linear_generalized_maxwell,  ///< global indicator for QLV Maxwell model
          bool& visco_fsls                               ///< global indicator for FSLS model
          ) override
      {
        visco_iso_rate = true;
        return;
      };


     private:
      /// my material parameters
      Mat::ViscoElast::PAR::CoupMyocard* params_;
    };

  }  // namespace ViscoElast
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_VISCO_ISORATEDEP_HPP
#define FOUR_C_MAT_ELAST_VISCO_ISORATEDEP_HPP

#include "4C_config.hpp"

#include "4C_mat_elast_summand.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Elastic
  {
    namespace PAR
    {
      /*!
       * @brief material parameters for isochoric contribution of a frequency independent
       * viscoelastic material
       *
       * <h3>Input line</h3>
       * MAT 1 VISCO_IsoRateDep N 1
       */
      class IsoRateDep : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoRateDep(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// material parameters
        double n_;

        //@}

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        std::shared_ptr<Core::Mat::Material> create_material() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "Mat::Elastic::Summand::Factory.");
          return nullptr;
        };
      };  // class IsoRateDep

    }  // namespace PAR

    /*!
     * @brief Isochoric general power material
     *
     * This is the isochoric part of a viscohyperelastic, isotropic
     * material depending on the modified invariants of the rate of the right
     * Cauchy-Green tensor.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = n \overline{J}_2 (\overline{I}_1 -3).
     * \f]
     * (n = \eta)
     */
    class IsoRateDep : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoRateDep(Mat::Elastic::PAR::IsoRateDep* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isoratedep;
      }

      //@}

      /// Add modified coeffiencts.
      void add_coefficients_visco_modified(
          const Core::LinAlg::Matrix<3, 1>&
              modinv,                         ///< modified invariants of right Cauchy-Green tensor
          Core::LinAlg::Matrix<8, 1>& modmu,  ///< necessary coefficients for piola-kirchhoff-stress
          Core::LinAlg::Matrix<33, 1>& modxi,  ///< necessary coefficients for viscosity tensor
          Core::LinAlg::Matrix<7, 1>& modrateinv, const Teuchos::ParameterList& params, int gp,
          int eleGID) override;

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
        isomod = true;
        viscogeneral = true;
        return;
      };

      /// Indicator for the chosen viscoelastic formulations
      void specify_visco_formulation(
          bool& visco_iso_rate,  ///< global indicator for isotropic rate-dependent visco response
          bool& visco_legacy_sls_removed,   ///< global indicator for removed legacy SLS model
          bool& visco_generalized_maxwell,  ///< global indicator for generalized Maxwell model
          bool& visco_fsls                  ///< global indicator for FSLS model
          ) override
      {
        visco_iso_rate = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::IsoRateDep* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

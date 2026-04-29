// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_VISCO_FSLS_HPP
#define FOUR_C_MAT_ELAST_VISCO_FSLS_HPP

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
       * @brief material parameters for viscous contribution according the FSLS-model
       *
       * <h3>Input line</h3>
       * MAT 1 VISCO_FSLS TAU 0.1 ALPHA 0.5 BETA 1
       */
      class Fsls : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        Fsls(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// material parameters
        double tau_;
        double alpha_;
        double beta_;

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
      };  // class Fsls
    }  // namespace PAR


    /*!
     * @brief Material VISCO_FSLS
     *
     * This material offers a viscous and hyperelastic part. The model consists
     * of one spring in parallel to one sequential branch of a spring and a springpot.
     *
     * A springpot is between a spring and a dashpot. The parameter alpha regulates
     * how much damping is introduced.
     * Alpha=0, means the springpot is a spring
     * Alpha=1, means the springpot is a dashpot; this is equal to a generalized Maxwell branch
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] Adolfson and Enelund (2003): Fractional Derivative Viscoelasticity at
     *          Large Deformations
     * </ul>
     */
    class Fsls : public Summand
    {
     public:
      /// constructor with given material parameters
      Fsls(Mat::Elastic::PAR::Fsls* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_fsls;
      }

      //@}

      /// Read material parameters
      void read_material_parameters_visco(double& tau,  ///< relaxation parameter tau
          double& beta,                                 ///< emphasis of viscous to elastic part
          double& alpha,                                ///< fractional order derivative (for FSLS)
          std::string& solve  ///< unused for FSLS; kept for interface compatibility
          ) override;

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
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
        visco_fsls = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::Fsls* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

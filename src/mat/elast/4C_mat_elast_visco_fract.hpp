// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_VISCO_FRACT_HPP
#define FOUR_C_MAT_ELAST_VISCO_FRACT_HPP

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
      class Fract : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        Fract(const Core::Mat::PAR::Parameter::Data& matdata);

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
      };  // class Fract
    }  // namespace PAR


    /*!
     * @brief Material Viscofract
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
    class Fract : public Summand
    {
     public:
      /// constructor with given material parameters
      Fract(Mat::Elastic::PAR::Fract* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_fract;
      }

      //@}

      /// Read material parameters
      void read_material_parameters_visco(double& tau,  ///< relaxation parameter tau
          double& beta,                                 ///< emphasis of viscous to elastic part
          double& alpha,  ///< fractional order derivative (just for visoc_fract)
          std::string&
              solve  //!< variant of the solution of the evolution integral (just for genmax)
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
          bool& isovisco,     ///< global indicator for isotropic, split and viscous formulation
          bool& viscogenmax,  ///< global indicator for viscous contribution according the SLS-Model
          bool& viscogeneralizedgenmax,  ///< global indicator for viscoelastic contribution
                                         ///< according to the generalized Maxwell Model
          bool& viscofract  ///< global indicator for viscous contribution according the FSLS-Model
          ) override
      {
        viscofract = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::Fract* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

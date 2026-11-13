// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISOTESTMATERIAL_HPP
#define FOUR_C_MAT_ELAST_ISOTESTMATERIAL_HPP

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
       * @brief material parameters for isochoric contribution of test material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_IsoTestMaterial C1 100 C2 50
       */
      class IsoTestMaterial : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoTestMaterial(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Shear modulus
        double c1_;
        double c2_;

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
      };  // class IsoTestMaterial
    }  // namespace PAR

    /*!
     * @brief Isochoric Material to test the elasthyper-Toolbox
     *
     * This material is not realistic, but contains all possible derivatives of invariants. With
     * this material in combination with volsussmanbathe, it is possible to test all isochoric
     * parts of the Elasthyper-Toolbox.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = C1 (\overline{I}_{\boldsymbol{C}}-3) + 0.5 C1 (\overline{I}_{\boldsymbol{C}}-3)^2
     *        + C2 (\overline{II}_{\boldsymbol{C}}-3)  + 0.5 C2
     *        (\overline{II}_{\boldsymbol{C}}-3)^2
     *        + D (\overline{I}_{\boldsymbol{C}}-3) (\overline{II}_{\boldsymbol{C}}-3).
     * \f]
     *
     * with D = C1 + 2 C2
     */
    class IsoTestMaterial : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoTestMaterial(Mat::Elastic::PAR::IsoTestMaterial* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isotestmaterial;
      }

      //@}

      // add strain energy
      void add_strain_energy(double& psi,  ///< strain energy function
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,  ///< Green-Lagrange strain
          int gp,                                                       ///< Gauss point
          const int eleGID                                              ///< element GID
          ) override;

      void add_derivatives_modified(
          Core::LinAlg::Matrix<3, 1>&
              dPmodI,  ///< first derivative with respect to modified invariants
          Core::LinAlg::Matrix<6, 1>&
              ddPmodII,  ///< second derivative with respect to modified invariants
          const Core::LinAlg::Matrix<3, 1>&
              modinv,       ///< modified invariants of right Cauchy-Green tensor
          int gp,           ///< Gauss point
          const int eleGID  ///< element GID
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
        isomod = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::IsoTestMaterial* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUP3POW_HPP
#define FOUR_C_MAT_ELAST_COUP3POW_HPP

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
       * @brief material parameters for volumtric contribution of a general power material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_Coup3Pow C 1 D 1
       */
      class Coup3Pow : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        Coup3Pow(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// material parameters
        double c_;
        int d_;

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
      };  // class Coup3Pow
    }  // namespace PAR

    /*!
     * @brief Volumtric general power material
     *
     * This is a summand of a variable order hyperelastic, volumetric
     * material depending on the first invariant of the right Cauchy-Green tensor.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = C (I_3^(1/3)-1)^D.
     * \f]
     */
    class Coup3Pow : public Summand
    {
     public:
      /// constructor with given material parameters
      Coup3Pow(Mat::Elastic::PAR::Coup3Pow* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coup3pow;
      }

      //@}

      // add strain energy
      void add_strain_energy(double& psi,  ///< strain energy function
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,  ///< Green-Lagrange strain
          const int gp,
          const int eleGID  ///< element GID
          ) override;

      void add_derivatives_principal(
          Core::LinAlg::Matrix<3, 1>& dPI,    ///< first derivative with respect to invariants
          Core::LinAlg::Matrix<6, 1>& ddPII,  ///< second derivative with respect to invariants
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          int gp,     ///< Gauss point
          int eleGID  ///< element GID
          ) override;

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< global indicator, if one viscoelastic formulation is used
          ) override
      {
        isoprinc = true;
        return;
      };


     private:
      /// my material parameters
      Mat::Elastic::PAR::Coup3Pow* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

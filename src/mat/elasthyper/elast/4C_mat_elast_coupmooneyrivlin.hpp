// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPMOONEYRIVLIN_HPP
#define FOUR_C_MAT_ELAST_COUPMOONEYRIVLIN_HPP

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
       * @brief material parameters for coupled contribution of a CoupMooneyRivlinan material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupMooneyRivlin C1 1 C2 1 C3 1
       */
      class CoupMooneyRivlin : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupMooneyRivlin(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// material parameters
        double c1_;
        double c2_;
        double c3_;
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
      };  // class CoupMooneyRivlin

    }  // namespace PAR

    /*!
     * @brief CoupMooneyRivlinan material
     *
     * Mooney-Rivlin type nearly incompressible, hyperelastic 3D material law.
     *
     * The underlying strain-energy function is (expressed in invariants I1 and I2):
     * \f[
     *     \Psi= c_1 (I1 - 3)  +  c_2 (I2 - 3)  -  (2 c_1 + 4 c_2) ln(J) + c_3 * (J - 1)^2
     * \f]
     *
     * For references see Holzapfel p. 245
     *
     * Parameters are \f$ c_1, c_2\f$ and \f$c_3\f$ as penalty factor to enforce incompressibility
     * (shear modulus \f$\mu = (c_1 + c_2) / 2)\f$
     */
    class CoupMooneyRivlin : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupMooneyRivlin(Mat::Elastic::PAR::CoupMooneyRivlin* params);


      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupmooneyrivlin;
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
          int eleGID                                                    ///< element GID
          ) override;

      void add_derivatives_principal(
          Core::LinAlg::Matrix<3, 1>& dPI,    ///< first derivative with respect to invariants
          Core::LinAlg::Matrix<6, 1>& ddPII,  ///< second derivative with respect to invariants
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          int gp,     ///< Gauss point
          int eleGID  ///< element GID
          ) override;

      /// add the derivatives of a coupled strain energy functions associated with a purely
      /// isochoric deformation
      void add_coup_deriv_vol(
          const double j, double* dPj1, double* dPj2, double* dPj3, double* dPj4) override;

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
      Mat::Elastic::PAR::CoupMooneyRivlin* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPEXPPOL_HPP
#define FOUR_C_MAT_ELAST_COUPEXPPOL_HPP

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
       * @brief material parameters for compressible soft tissue material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupExpPol A 600. B 2. C 5.
       */
      class CoupExpPol : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupExpPol(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{
        double a_;
        /// constant for linear part of I_1
        double b_;
        /// constant for linear part of J
        double c_;
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
      };  // class CoupExpPol

    }  // namespace PAR

    /*!
     * @brief coupled hyperelastic, compressible, isotropic material according to [1] with linear
     * summands in exponent The material strain energy density function is
     * \f[
     *    \Psi = a \exp[ b(I_1- 3) - (2b + c)ln{J} + c(J-1) ] - a
     * \f]
     *
     * More details at #AddCoefficientsPrincipal()
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] Weickenmeier, Jabareen "Elastic-viscoplastic modeling of soft biological
     *           tissues using a mixed finite element formulation based on the relative
     *           deformation gradient", 2014
     * </ul>
     */
    class CoupExpPol : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupExpPol(Mat::Elastic::PAR::CoupExpPol* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupexppol;
      }

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

      // add first and second derivative w.r.t. principal invariants
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
      Mat::Elastic::PAR::CoupExpPol* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPVARGA_HPP
#define FOUR_C_MAT_ELAST_COUPVARGA_HPP

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
       * @brief material parameters of Varga's material
       *
       *  <h3>Input line</h3>
       *  MAT 1 ELAST_CoupVarga MUE 1.0 BETA 1.0
       */
      class CoupVarga : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupVarga(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Shear modulus
        double mue_;
        /// 'Anti-modulus'
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

      };  // class CoupVarga
    }  // namespace PAR

    /*!
     * @brief Varga's material according to [1], [2]
     *
     *  This is a compressible, hyperelastic material
     *  of the most simple kind.
     *
     *  The material strain energy density function is
     * \f[
     *  \Psi = \underbrace{(2\mu-\beta)}_{\displaystyle\alpha} \Big(\lambda_1 + \lambda_2 +
     *  \lambda_3 - 3\Big) + \beta \Big(\frac{1}{\lambda_1} + \frac{1}{\lambda_2} +
     * \frac{1}{\lambda_3} - 3\Big) \f] which was taken from [1] Eq (6.129) and [2] Eq (1.3).
     *
     *  The material is stress-free in the \f$\lambda_1=\lambda_2=\lambda_3=1\f$ configuration
     *  if \f$\beta=\alpha=\mu\f$.
     *
     *  <h3>References</h3>
     *  <ul>
     *  <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
     *  <li> [2] JM Hill and DJ Arrigo, "New families of exact solutions for
     *           finitely deformed incompressible elastic materials",
     *           IMA J Appl Math, 54:109-123, 1995.
     *  </ul>
     */
    class CoupVarga : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupVarga(Mat::Elastic::PAR::CoupVarga* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupvarga;
      }

      /// add shear modulus equivalent
      void add_shear_mod(bool& haveshearmod,  ///< non-zero shear modulus was added
          double& shearmod,                   ///< variable to add upon
          int ele_gid                         ///< element GID
      ) const override
      {
        // indeed, a shear modulus is provided
        haveshearmod = true;

        // material parameters for isochoric part
        shearmod += params_->mue_;
      };

      //@}

      /// Answer if coefficients with respect to principal stretches are provided
      bool have_coefficients_stretches_principal() override { return true; }


      /// Add coefficients with respect to principal stretches (or zeros)
      void add_coefficients_stretches_principal(
          Core::LinAlg::Matrix<3, 1>& gamma,  ///< see above, [gamma_1, gamma_2, gamma_3]
          Core::LinAlg::Matrix<6, 1>&
              delta,  ///< see above, [delta_11, delta_22, delta_33, delta_12, delta_23, delta_31]
          const Core::LinAlg::Matrix<3, 1>&
              prstr  ///< principal stretches, [lambda_1, lambda_2, lambda_3]
          ) override;

      /// add the derivatives of a coupled strain energy functions associated with a purely
      /// isochoric deformation
      void add_coup_deriv_vol(
          const double j, double* dPj1, double* dPj2, double* dPj3, double* dPj4) override
      {
        FOUR_C_THROW("not implemented");
      }

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< global indicator, if one viscoelastic formulation is used
          ) override
      {
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupVarga* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPLOGNEOHOOKE_HPP
#define FOUR_C_MAT_ELAST_COUPLOGNEOHOOKE_HPP

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
       * @brief material parameters for logarithmic neo-Hooke material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupLogNeoHooke MODE YN C1 1.0 C2 0.3
       * MAT 1 ELAST_CoupLogNeoHooke MODE Lame C1 1.0 C2 1.0
       */
      class CoupLogNeoHooke : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupLogNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Shear modulus
        double mue_;
        /// Lame's constant
        double lambda_;

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
      };  // class CoupLogNeoHooke

    }  // namespace PAR

    /*!
     * @brief Logarithmic neo-Hooke material according to [1]
     *
     * This is a compressible, hyperelastic, isotropic material
     * of the most simple kind.
     *
     * The material strain energy density function is
     * \f[
     * \Psi = \frac{\mu}{2} (I_{\boldsymbol{C}} - 3)
     *       - \mu \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})
     *      + \frac{\lambda}{2} \big( \log(\sqrt{I\!I\!I_{\boldsymbol{C}}}) \big)^2
     * \f]
     * which was taken from [1]. More details at #AddCoefficientsPrincipal()
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] J Bonet and RD Wood, "Nonlinear continuum mechanics for finite
     *          element analysis", Cambridge, 1997.
     * <li> [2] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
     * </ul>
     */
    class CoupLogNeoHooke : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupLogNeoHooke(Mat::Elastic::PAR::CoupLogNeoHooke* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_couplogneohooke;
      }

      /// add shear modulus equivalent
      void add_shear_mod(bool& haveshearmod,  ///< non-zero shear modulus was added
          double& shearmod,                   ///< variable to add upon
          int ele_gid                         ///< element GID
      ) const override
      {
        haveshearmod = true;
        shearmod += params_->mue_;
      };

      //@}

      /*!
       * @brief Main material call to determine  2nd PK stress and material constitutive tensor
       *
       * The material strain energy density function is
       * \f[
       * \Psi = \frac{\mu}{2} (I_{\boldsymbol{C}} - 3)
       *       - \mu \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})
       *      + \frac{\lambda}{2} \big( \log(\sqrt{I\!I\!I_{\boldsymbol{C}}}) \big)^2
       * \f]
       * which was taken from [1].
       *
       * Here is \f$I_{\boldsymbol{C}}\f$ the first principal invariant
       * of the right Cauchy--Green strain tensor \f$\boldsymbol{C}\f$
       * and \f$I\!I\!I_{\boldsymbol{C}}\f$ its third.
       * The isochoric part is proportional to \f$\mu\f$,
       * whereas the \f$\lambda\f$-proportional part constitutes the volumetric contribution.
       *
       * The parameters are the shear modulus. In #params_->parmode_==0 it is set directly,
       * in #params_->parmode_==1 it is computed by
       * \f[
       * \mu = \frac{E}{2(1+\nu)}
       * \f]
       * and Lame's coefficient, again for #params_->parmode_==1, otherwise set by user,
       * \f[
       * \lambda = \left\{\begin{array}{ll}
       *      \frac{E \nu}{(1+\nu) (1-2\nu)} & \text{if $\nu \neq 1/2$}
       *   \\ 0                              & \text{else}
       * \end{array}\right.
       * \f]
       *
       * The 2nd Piola--Kirchhoff stress is
       * \f[
       * \boldsymbol{S} = \mu \big( \boldsymbol{1} - \boldsymbol{C}^{-1} \big)
       *                + \lambda \, \log(\sqrt{I\!I\!I_{\boldsymbol{C}}}) \, \boldsymbol{C}^{-1}
       * \f]
       *
       * The material constitutive 4-tensor
       * \f$\boldsymbol{C}_\text{m}=C_{IJKL}\boldsymbol{E}^I\otimes\boldsymbol{E}^J\otimes\boldsymbol{E}^K\otimes\boldsymbol{E}^L\f$
       * is determined by
       * \f[
       * C_{IJKL}
       * = \lambda (\boldsymbol{C}^{-1})_{IJ} \, (\boldsymbol{C}^{-1})_{KL}
       * + 2\big(\mu-\lambda \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})\big)
       *   \, \frac{1}{2} \big( \boldsymbol{C}^{-1})_{IK} \, (\boldsymbol{C}^{-1})_{JL}
       *                        + (\boldsymbol{C}^{-1})_{IL} \, (\boldsymbol{C}^{-1})_{JK} \big)
       * \f]
       * or
       * \f[
       * \boldsymbol{C}_\text{m}
       * = \lambda \boldsymbol{C}^{-1} \otimes  \boldsymbol{C}^{-1}
       * + 2\big(\mu-\lambda \log(\sqrt{I\!I\!I_{\boldsymbol{C}}})\big)
       *   \, \boldsymbol{C}^{-1} \odot \boldsymbol{C}^{-1}
       * \f]
       */

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
        isoprinc = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupLogNeoHooke* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

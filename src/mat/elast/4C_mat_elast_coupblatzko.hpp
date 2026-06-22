// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPBLATZKO_HPP
#define FOUR_C_MAT_ELAST_COUPBLATZKO_HPP

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
       * @brief material parameters for Blatz and Ko material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupBlatzKo MUE 1.044E7 NUE 0.3 F 0.5
       */
      class CoupBlatzKo : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupBlatzKo(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Shear modulus
        double mue_;
        /// Possion's ratio
        double nue_;
        /// interpolation parameter
        double f_;

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
      };  // class CoupBlatzKo

    }  // namespace PAR
    /*!
     * @brief Coupled Blatz and Ko material according to [1].
     *
     * Strain energy function is given by
     * \f[
     *   \Psi= f \frac {\mu} 2 \left[ (I_{\boldsymbol C}-3)+\frac 1
     *         {\beta} ( III_{\boldsymbol C}^{-\beta} -1) \right]
     *         +(1-f) \frac {\mu} 2 \left[\left( \frac {II_{\boldsymbol
     *         C}}{III_{\boldsymbol C}}-3 \right) + \frac 1 {\beta}
     *         (III_{\boldsymbol C}^{\beta}-1)\right]
     * \f]
     *
     * \f[
     *    with \ \beta = \ nue/(1. - 2.*nue)
     * \f]
     *
     * [1] Holzapfel, G.A. "Nonlinear Solid Mechanics", 2000, p.247
     */
    class CoupBlatzKo : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupBlatzKo(Mat::Elastic::PAR::CoupBlatzKo* params);

      /// @name Access material constants
      //@{
      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupblatzko;
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

      // add first and second derivative w.r.t. principal invariants
      void add_derivatives_principal(
          Core::LinAlg::Matrix<3, 1>& dPI,    ///< first derivative with respect to invariants
          Core::LinAlg::Matrix<6, 1>& ddPII,  ///< second derivative with respect to invariants
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          int gp,     ///< Gauss point
          int eleGID  ///< element GID
          ) override;

      // add third derivative w.r.t. principal invariants
      void add_third_derivatives_principal_iso(
          Core::LinAlg::Matrix<10, 1>&
              dddPIII_iso,  ///< third derivative with respect to invariants
          const Core::LinAlg::Matrix<3, 1>& prinv_iso,  ///< principal isotropic invariants
          int gp,                                       ///< Gauss point
          int eleGID) override;                         ///< element GID

      /// add the derivatives of a coupled strain energy functions associated with a purely
      /// isochoric deformation
      void add_coup_deriv_vol(
          const double j, double* dPj1, double* dPj2, double* dPj3, double* dPj4) override;

      /// @name Access methods
      //@{
      double mue() const { return params_->mue_; }
      double nue() const { return params_->nue_; }
      //@}

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
      Mat::Elastic::PAR::CoupBlatzKo* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

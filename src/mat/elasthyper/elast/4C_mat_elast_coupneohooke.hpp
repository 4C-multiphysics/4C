// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPNEOHOOKE_HPP
#define FOUR_C_MAT_ELAST_COUPNEOHOOKE_HPP

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
       * @brief material parameters for isochoric contribution of a CoupNeoHookean material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupNeoHooke YOUNG 1 NUE 1
       */
      class CoupNeoHooke : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Young's modulus
        double youngs_;
        /// Possion's ratio
        double nue_;

        /// nue \(1-2 nue)
        double beta_;
        /// Shear modulus / 2
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
      };  // class CoupNeoHooke

    }  // namespace PAR

    /*!
     * @brief CoupNeoHookean material
     *
     * This is the summand of a hyperelastic, isotropic CoupNeoHookean material depending on the
     * first and the third invariant of the right Cauchy-Green tensor. The formulation is based on
     * [1] p. 247,248 and 263
     *
     * The implemented material is the coupled form of the compressible NeoHooke model. The
     * Parameters read in are the Young's modulus and the Poisson's ratio.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = c(I_{\boldsymbol{C}}-3)+\frac {c}{\beta} (J^{-2\beta}-1)
     * \f]
     *
     * with
     * \f[
     *   \beta = \frac {\nu}{1-2\nu}
     * \f]
     *
     * \f$ c=\frac {\mu}{2} = \frac {\text{Young's modulus}}{4(1+\nu)} \f$ and \f$\mu\f$ and
     * \f$\nu\f$ denoting the shear modulus and the Poisson's ratio, respectively.
     *
     * [1] Holzapfel, G. A., Nonlinear Solid Mechanics, 2002
     */
    class CoupNeoHooke : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupNeoHooke(Mat::Elastic::PAR::CoupNeoHooke* params);


      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupneohooke;
      }

      /// add shear modulus equivalent
      void add_shear_mod(bool& haveshearmod,  ///< non-zero shear modulus was added
          double& shearmod,                   ///< variable to add upon
          int ele_gid                         ///< element GID
      ) const override
      {
        haveshearmod = true;
        shearmod += 2 * params_->c_;
      };

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
      double nue() const { return params_->nue_; }
      double youngs() const { return params_->youngs_; }
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
      Mat::Elastic::PAR::CoupNeoHooke* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

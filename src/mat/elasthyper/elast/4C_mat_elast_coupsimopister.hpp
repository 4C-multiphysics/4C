// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPSIMOPISTER_HPP
#define FOUR_C_MAT_ELAST_COUPSIMOPISTER_HPP

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
       * @brief material parameters for isochoric contribution of a CoupSimoPisteran material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupSimoPister MUE 1000
       */
      class CoupSimoPister : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        CoupSimoPister(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{
        double mue_;
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
      };  // class CoupSimoPister

    }  // namespace PAR

    /*!
     * @brief CoupSimoPisteran material
     *
     * This is the summand of a hyperelastic, isotropic CoupSimoPisteran material
     * depending on the first and the third invariant of the right Cauchy-Green tensor.
     * The formulation is based on [1] (4.3) or [2] (1)
     *
     * The implemented material is a coupled form of the Simo & Pister material
     * model. The part U(J) is not implemented. So this material model can/should
     * combined with any volumetric material model.
     * The Parameter read in is my.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = 0.5*\mu(I_{\boldsymbol{C}}-3) - \mu log(J)
     * \f]
     *
     * [1] Simo and Pister - 1984
     * [2] Hartmann - "The class of Simo & Pister-type Hyperelasticity Relations"
     */
    class CoupSimoPister : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupSimoPister(Mat::Elastic::PAR::CoupSimoPister* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupsimopister;
      }

      //@}

      // add strain energy
      void add_strain_energy(double& psi,  ///< strain energy function
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              glstrain,  ///< Green-Lagrange strain in strain like Voigt notation
          int gp,        ///< Gauss point
          int eleGID     ///< element GID
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
      Mat::Elastic::PAR::CoupSimoPister* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

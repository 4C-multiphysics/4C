// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_VOLPENALTY_HPP
#define FOUR_C_MAT_ELAST_VOLPENALTY_HPP

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
       * @brief material parameters for volumetric contribution
       * \f$\Psi=\epsilon \left( J^{\gamma} + \frac 1 {J^{\gamma}} -2 \right)\f$
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_VolPenalty EPSILON 1. GAMMA 1
       */
      class VolPenalty : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        VolPenalty(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Dilatation modulus
        double eps_;
        /// empiric constant
        double game_;

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
      };  // class VolPenalty

    }  // namespace PAR

    /*!
     * @brief Volumetric Penalty
     *
     * Strain energy function is given by
     * \f[
     *   \Psi=\epsilon \left( J^{\gamma} + \frac 1 {J^{\gamma}} -2 \right)
     * \f]
     */
    class VolPenalty : public Summand
    {
     public:
      /// constructor with given material parameters
      VolPenalty(Mat::Elastic::PAR::VolPenalty* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_volpenalty;
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

      // Add derivatives with respect to modified invariants.
      void add_derivatives_modified(
          Core::LinAlg::Matrix<3, 1>&
              dPmodI,  ///< first derivative with respect to modified invariants
          Core::LinAlg::Matrix<6, 1>&
              ddPmodII,  ///< second derivative with respect to modified invariants
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// Add third derivative w.r.t. J
      void add3rd_vol_deriv(const Core::LinAlg::Matrix<3, 1>& modinv, double& d3PsiVolDJ3) override;

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
      Mat::Elastic::PAR::VolPenalty* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

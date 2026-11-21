// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISOVARGA_HPP
#define FOUR_C_MAT_ELAST_ISOVARGA_HPP

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
       * <h3>Input line</h3>
       * MAT 1 ELAST_IsoVarga MUE 1.0 BETA 1.0
       */
      class IsoVarga : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoVarga(const Core::Mat::PAR::Parameter::Data& matdata);

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

      };  // class IsoVarga

    }  // namespace PAR

    /*!
     * \brief Isochoric Varga's material according to [1], [2]
     *
     * This is a compressible, hyperelastic, isotropic material
     * of the most simple kind.
     *
     * The material strain energy density function is
     * \f[
     * \Psi = \underbrace{(2\mu-\beta)}_{\displaystyle\alpha} \Big(\bar{\lambda}_1 +
     * \bar{\lambda}_2 + \bar{\lambda}_3 - 3\Big)
     *      + \beta \Big(\frac{1}{\bar{\lambda}_1} + \frac{1}{\bar{\lambda}_2} +
     *      \frac{1}{\bar{\lambda}_3} - 3\Big)
     * \f]
     * which was taken from [1] Eq (6.129) and [2] Eq (1.3).
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
     * <li> [2] JM Hill and DJ Arrigo, "New families of exact solutions for
     *          finitely deformed incompressible elastic materials",
     *          IMA J Appl Math, 54:109-123, 1995.
     * </ul>
     */
    class IsoVarga : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoVarga(Mat::Elastic::PAR::IsoVarga* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isovarga;
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
      bool have_coefficients_stretches_modified() override { return true; }

      /// Add coefficients with respect to modified principal stretches (or zeros)
      void add_coefficients_stretches_modified(
          Core::LinAlg::Matrix<3, 1>&
              modgamma,  ///< [\bar{\gamma}_1, \bar{\gamma}_2, \bar{\gamma}_3]
          Core::LinAlg::Matrix<6, 1>&
              moddelta,  ///< [\bar{\delta}_11, \bar{\delta}_22, \bar{\delta}_33,
                         ///< \bar{\delta}_12,\bar{\delta}_23, \bar{\delta}_31]
          const Core::LinAlg::Matrix<3, 1>&
              modstr  ///< modified principal stretches, [\bar{\lambda}_1,
                      ///< \bar{\lambda}_2, \bar{\lambda}_3]
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
        return;
      };

     private:
      /// Varga material parameters
      Mat::Elastic::PAR::IsoVarga* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

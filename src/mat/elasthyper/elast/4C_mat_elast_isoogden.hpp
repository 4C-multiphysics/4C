// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISOOGDEN_HPP
#define FOUR_C_MAT_ELAST_ISOOGDEN_HPP

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
      class IsoOgden : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoOgden(const Core::Mat::PAR::Parameter::Data& matdata);

        //! @name material parameters
        //! @{
        double mue_;    ///< shear modulus
        double alpha_;  ///< nonlinearity parameter
        //! @}

        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        std::shared_ptr<Core::Mat::Material> create_material() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "Mat::Elastic::Summand::Factory.");
          return nullptr;
        };
      };  // class IsoOgden
    }  // namespace PAR

    /*!
     * \brief Isochoric part of the one-term Ogden material, see Holzapfel [1] or Ogden [2]
     *
     * This is a hyperelastic isotropic material with compression-tension asymmetry expressed in
     * terms of the modified principal stretches. Amongst other applications, it can be used to
     * model human brain tissue [3]. In contrast to the fully incompressible formulation in [1],
     * here, the compressible formulation, i.e. the isochoric contribution of a decoupled
     * strain-energy function is implemented. Further, here the number of considered terms N is set
     * to one, s.t. the formulation reduces to a one-term Ogden model.
     *
     * The strain-energy function is hence given in terms of the modified principal stretches as:
     * \f[
     *   \Psi_{iso}=\frac{2\mu}{\alpha^2}\,(\bar{\lambda}_1^\alpha+\bar{\lambda}_2^\alpha+\bar{\lambda}_3^\alpha-3)
     * \f]
     *
     * References:
     * [1] G. A. Holzapfel, 'Nonlinear solid mechanics', Wiley, pp. 235-236, 2000.
     * [2] R. W. Ogden, 'Large deformation isotropic elasticity: on the correlation of theory and
     * experiment for compressible rubberlike solids', Proc. R. Soc. Long. A, vol. 326, pp. 565-584,
     * 1972, doi: 10.1098/rspa.1972.0096.
     * [3] S. Budday et. al., 'Mechanical characterization of human brain tissue', Acta
     * Biomaterialia, vol. 48, pp. 319-340, 2017, doi: 10.1016/j.actbio.2016.10.036.
     */
    class IsoOgden : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoOgden(Mat::Elastic::PAR::IsoOgden* params);

      /// Provide the material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isoogden;
      }

      /// Answer if coefficients with respect to modified principal stretches are provided
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

      /// Specify the formulation as isochoric in terms of modified principal invariants
      void specify_formulation(bool& isoprinc, bool& isomod, bool& anisoprinc, bool& anisomod,
          bool& viscogeneral) override
      {
        isomod = true;
      };

     private:
      /// one-term Ogden material parameters
      Mat::Elastic::PAR::IsoOgden* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
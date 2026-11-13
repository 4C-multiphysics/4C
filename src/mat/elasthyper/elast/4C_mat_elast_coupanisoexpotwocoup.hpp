// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPANISOEXPOTWOCOUP_HPP
#define FOUR_C_MAT_ELAST_COUPANISOEXPOTWOCOUP_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_par_aniso.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Elastic
  {
    namespace PAR
    {
      /*!
       * @brief material parameters for coupled passive cardiac material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupAnisoExpoTwoCoup A4 18.472 B4 16.026 A6 2.481 B6 11.120 A8 0.216 B8 11.436
       * GAMMA 0.0 [INIT 1] [FIB_COMP Yes] [ADAPT_ANGLE No]
       */
      class CoupAnisoExpoTwoCoup : public Mat::PAR::ParameterAniso
      {
       public:
        /// constructor with given material parameters
        explicit CoupAnisoExpoTwoCoup(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// fiber params
        double A4_;
        double B4_;
        double A6_;
        double B6_;
        double A8_;
        double B8_;
        /// angle between circumferential and fiber direction (used for cir, axi, rad nomenclature)
        double gamma_;
        /// fiber initialization status
        int init_;
        /// fibers support compression - or not
        bool fib_comp_;
        /// adapt angle during remodeling
        bool adapt_angle_;

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
      };  // class CoupAnisoExpoTwoCoup

    }  // namespace PAR

    /*!
     * \brief Anisotropy manager for two fibers and the structural tensor of the combination of
     * those two
     */
    class CoupAnisoExpoTwoCoupAnisoExtension : public DefaultAnisotropyExtension<2>
    {
     public:
      /*!
       * \brief Constructor
       *
       * \param params Material parameters
       */
      explicit CoupAnisoExpoTwoCoupAnisoExtension(Mat::Elastic::PAR::CoupAnisoExpoTwoCoup* params);

      /*!
       * \brief Pack all data for parallel distribution and restart
       *
       * \param data data array to pack to
       */
      void pack_anisotropy(Core::Communication::PackBuffer& data) const override;

      /*!
       * \brief Unpack data from the pack from parallel distribution and restart
       *
       * \param data data array to unpack from
       * \param position position of the data
       */
      void unpack_anisotropy(Core::Communication::UnpackBuffer& buffer) override;

      /*!
       * \brief Notifier method when fibers are initialized.
       */
      void on_fibers_initialized() override;

      /*!
       * \brief Returns the reference to the coupled structural tensor in stress like Voigt
       * notation
       *
       * \param gp Gauss point
       * \return const Core::LinAlg::Matrix<6, 1>& Reference to the coupled structural tensor in
       * stress like Voigt notation
       */
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_coupled_structural_tensor(
          int gp) const;

      /*!
       * \brief Returns the coupled scalar product at the Gauss point
       *
       * \param gp Gauss point
       * \return double Scalar product of the two fibers
       */
      double get_coupled_scalar_product(int gp) const;

     private:
      /// dot product fiber direction
      std::vector<double> a1a2_;

      /// mixed structural tensor (symmetric) \f$\frac{1}{2}(a1 \otimes a2 + a2 \otimes a1)\f$ in
      /// stress like Voigt notation
      std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> a1_a2_;
    };

    /*!
     * @brief Anisotropic cardiac material, implemented with two possible fiber families as in [1]
     *
     * This is a hyperelastic, anisotropic material for the passive response of cardiac material
     *
     * Strain energy function is given by:
     * \f[
     *   \Psi = \frac {a_4}{2 b_4} \left( e^{b_4 (IV_{\boldsymbol C} - 1)^2} - 1 \right) +
     *   \frac {a_6}{2 b_6} \left( e^{b_6 (VI_{\boldsymbol C} - 1)^2} - 1 \right) + \frac{a_8}{2b_8}
     * \left( e^{b_8 \left( VIII_{\boldsymbol C} - a_0 \cdot b_0 \right)^2} - 1\right) \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] GA Holzapfel, RW Ogden, Constitutive modelling of passive myocardium: a
     * structurally based framework for material characterization
     * <li> [2] C Sansour, On the physical assumptions underlying the volumetric-isochoric split
     * and the case of anisotropy
     * </ul>
     */
    class CoupAnisoExpoTwoCoup : public Summand
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpoTwoCoup(Mat::Elastic::PAR::CoupAnisoExpoTwoCoup* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupanisoexpotwocoup;
      }
      //@}

      /*!
       * \brief Register the local anisotropy extension to the global anisotropy manager
       *
       * \param anisotropy Reference to the global anisotropy manager
       */
      void register_anisotropy_extensions(Anisotropy& anisotropy) override;

      /// @name Methods for Packing and Unpacking
      ///@{
      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;
      ///@}

      /// Add anisotropic principal stresses
      void add_stress_aniso_principal(
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,   ///< right Cauchy Green Tensor
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,      ///< 2nd PK-stress
          const Teuchos::ParameterList&
              params,  ///< additional parameters for computation of material properties
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// Set fiber directions
      void set_fiber_vecs(double newgamma,                   ///< new angle
          const Core::LinAlg::Tensor<double, 3, 3>& locsys,  ///< local coordinate system
          const Core::LinAlg::Tensor<double, 3, 3>& defgrd   ///< deformation gradient
          ) override;

      /// Get fiber directions
      void get_fiber_vecs(
          std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
      ) const override;

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< global indicator, if one viscoelastic formulation is used
          ) override
      {
        anisoprinc = true;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupAnisoExpoTwoCoup* params_;

      /// Special anisotropic behavior
      CoupAnisoExpoTwoCoupAnisoExtension anisotropy_extension_;
    };  // namespace PAR

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPANISOEXPO_HPP
#define FOUR_C_MAT_ELAST_COUPANISOEXPO_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_elast_coupanisoexpobase.hpp"
#include "4C_mat_par_aniso.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace Elastic
  {
    /*!
     * \brief Container class for communication with the base material
     */
    class CoupAnisoExpoAnisotropyExtension : public DefaultAnisotropyExtension<1>,
                                             public CoupAnisoExpoBaseInterface
    {
     public:
      /*!
       * \brief Constructor
       *
       * \param init_mode initialization mode of the fibers
       * \param gamma Angle of the fiber if they are given in a local coordinate system
       * \param adapt_angle boolean, whether the fiber is subject to growth and remodeling
       * \param structuralTensorStrategy Strategy to compute the structural tensor
       * \param fiber_id Id of the fiber to be used for the fiber (0 for FIBER1)
       */
      CoupAnisoExpoAnisotropyExtension(int init_mode, double gamma, bool adapt_angle,
          const std::shared_ptr<Elastic::StructuralTensorStrategyBase>& structuralTensorStrategy,
          int fiber_id);

      /*!
       * \copydoc
       *
       * \note The scalar product between the same fiber is 1, so nothing needs to be computed here
       */
      double get_scalar_product(int gp) const override;

      /*!
       * \brief Returns the fiber at the Gauss point
       *
       * \param gp Gauss point
       * \return const Core::LinAlg::Matrix<3, 1>& Constant reference to the fiber
       */
      const Core::LinAlg::Tensor<double, 3>& get_fiber(int gp) const;
      const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_structural_tensor(
          int gp) const override;

      // Tell the compiler that we still want the methods from FiberAnisotropyExtension with a
      // different signature
      using FiberAnisotropyExtension<1>::get_fiber;
      using FiberAnisotropyExtension<1>::get_structural_tensor;
    };

    namespace PAR
    {
      /*!
       * @brief material parameters for coupled contribution of a anisotropic exponential fiber
       * material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupAnisoExpo K1 10.0 K2 1.0 GAMMA 35.0 K1COMP 0.0 K2COMP 1.0 INIT 0
       * ADAPT_ANGLE 0
       */
      class CoupAnisoExpo : public Mat::PAR::ParameterAniso,
                            public Mat::Elastic::PAR::CoupAnisoExpoBase
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpo(const Core::Mat::PAR::Parameter::Data& matdata);

        std::shared_ptr<Core::Mat::Material> create_material() override { return nullptr; };

        /// @name material parameters
        //@{
        /// adapt angle during remodeling
        bool adapt_angle_;

        /// Id of the fiber to be used
        const int fiber_id_;
        //@}

      };  // class CoupAnisoExpo

    }  // namespace PAR

    /*!
     * @brief Coupled anisotropic exponential fiber function, implemented for one possible fiber
     * family as in [1]
     *
     * This is a hyperelastic, anisotropic material
     * of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = \frac {k_1}{2 k_2} \left(e^{k_2 (IV_{\boldsymbol C}-1)^2 }-1 \right).
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] G.A. Holzapfel, T.C. Gasser, R.W. Ogden: A new constitutive framework for arterial
     * wall mechanics
     *          and a comparative study of material models, J. of Elasticity 61 (2000) 1-48.
     * </ul>
     */
    class CoupAnisoExpo : public Mat::Elastic::CoupAnisoExpoBase,
                          public FiberAnisotropyExtensionProvider<1>
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpo(Mat::Elastic::PAR::CoupAnisoExpo* params);

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupanisoexpo;
      }

      //@}

      /// @name Methods for Packing and Unpacking
      ///@{
      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;
      ///@}

      /*!
       * \brief Register the anisotropy extension to the global anisotropy manager
       *
       * \param anisotropy anisotropy manager
       */
      void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

      /// Set fiber directions
      void set_fiber_vecs(double newgamma,                   ///< new angle
          const Core::LinAlg::Tensor<double, 3, 3>& locsys,  ///< local coordinate system
          const Core::LinAlg::Tensor<double, 3, 3>& defgrd   ///< deformation gradient
          ) override;

      /// Set fiber directions
      void set_fiber_vecs(const Core::LinAlg::Tensor<double, 3>& fibervec  ///< new fiber vector
          ) override;

      /// Get fiber directions
      void get_fiber_vecs(
          std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
      ) const override;

      /*!
       * \brief Returns the reference to the Mat::FiberAnisotropyExtension
       *
       * \return FiberAnisotropyExtension& Reference to the used Mat::FiberAnisotropyExtension
       */
      FiberAnisotropyExtension<1>& get_fiber_anisotropy_extension() override
      {
        return anisotropy_extension_;
      }

     protected:
      const CoupAnisoExpoBaseInterface& get_coup_aniso_expo_base_interface() const override
      {
        return anisotropy_extension_;
      }

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupAnisoExpo* params_;

      /// Internal ansotropy information
      Mat::Elastic::CoupAnisoExpoAnisotropyExtension anisotropy_extension_;
    };  // namespace PAR

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

/*----------------------------------------------------------------------*/
/*! \file
\brief Definitions of classes for the coupled exponential summand for fibers

\level 1
*/
/*----------------------------------------------------------------------*/

#ifndef FOUR_C_MATELAST_COUPANISOEXPO_HPP
#define FOUR_C_MATELAST_COUPANISOEXPO_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_par_aniso.hpp"
#include "4C_matelast_coupanisoexpobase.hpp"

FOUR_C_NAMESPACE_OPEN


namespace MAT
{
  namespace ELASTIC
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
          const Teuchos::RCP<ELASTIC::StructuralTensorStrategyBase>& structuralTensorStrategy,
          int fiber_id);

      /*!
       * \copydoc
       *
       * \note The scalar product between the same fiber is 1, so nothing needs to be computed here
       */
      double GetScalarProduct(int gp) const override;

      /*!
       * \brief Returns the fiber at the Gauss point
       *
       * \param gp Gauss point
       * \return const CORE::LINALG::Matrix<3, 1>& Constant reference to the fiber
       */
      const CORE::LINALG::Matrix<3, 1>& GetFiber(int gp) const;
      const CORE::LINALG::Matrix<3, 3>& GetStructuralTensor(int gp) const override;
      const CORE::LINALG::Matrix<6, 1>& get_structural_tensor_stress(int gp) const override;

      // Tell the compiler that we still want the methods from FiberAnisotropyExtension with a
      // different signature
      using FiberAnisotropyExtension<1>::GetFiber;
      using FiberAnisotropyExtension<1>::GetStructuralTensor;
      using FiberAnisotropyExtension<1>::get_structural_tensor_stress;
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
      class CoupAnisoExpo : public MAT::PAR::ParameterAniso,
                            public MAT::ELASTIC::PAR::CoupAnisoExpoBase
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpo(const Teuchos::RCP<CORE::MAT::PAR::Material>& matdata);

        Teuchos::RCP<CORE::MAT::Material> create_material() override { return Teuchos::null; };

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
    class CoupAnisoExpo : public MAT::ELASTIC::CoupAnisoExpoBase,
                          public FiberAnisotropyExtensionProvider<1>
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpo(MAT::ELASTIC::PAR::CoupAnisoExpo* params);

      /// @name Access material constants
      //@{

      /// material type
      CORE::Materials::MaterialType MaterialType() const override
      {
        return CORE::Materials::mes_coupanisoexpo;
      }

      //@}

      /// @name Methods for Packing and Unpacking
      ///@{
      void PackSummand(CORE::COMM::PackBuffer& data) const override;

      void UnpackSummand(
          const std::vector<char>& data, std::vector<char>::size_type& position) override;
      ///@}

      /*!
       * \brief Register the anisotropy extension to the global anisotropy manager
       *
       * \param anisotropy anisotropy manager
       */
      void register_anisotropy_extensions(MAT::Anisotropy& anisotropy) override;

      /// Set fiber directions
      void SetFiberVecs(double newgamma,             ///< new angle
          const CORE::LINALG::Matrix<3, 3>& locsys,  ///< local coordinate system
          const CORE::LINALG::Matrix<3, 3>& defgrd   ///< deformation gradient
          ) override;

      /// Set fiber directions
      void SetFiberVecs(const CORE::LINALG::Matrix<3, 1>& fibervec  ///< new fiber vector
          ) override;

      /// Get fiber directions
      void GetFiberVecs(
          std::vector<CORE::LINALG::Matrix<3, 1>>& fibervecs  ///< vector of all fiber vectors
          ) override;

      /*!
       * \brief Returns the reference to the MAT::FiberAnisotropyExtension
       *
       * \return FiberAnisotropyExtension& Reference to the used MAT::FiberAnisotropyExtension
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
      MAT::ELASTIC::PAR::CoupAnisoExpo* params_;

      /// Internal ansotropy information
      MAT::ELASTIC::CoupAnisoExpoAnisotropyExtension anisotropy_extension_;
    };  // namespace PAR

  }  // namespace ELASTIC
}  // namespace MAT

FOUR_C_NAMESPACE_CLOSE

#endif
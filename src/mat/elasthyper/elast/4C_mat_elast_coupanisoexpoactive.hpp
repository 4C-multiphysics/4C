// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPANISOEXPOACTIVE_HPP
#define FOUR_C_MAT_ELAST_COUPANISOEXPOACTIVE_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_elast_activesummand.hpp"
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
       * @brief material parameters for coupled contribution of an anisotropic active fiber material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_CoupAnisoExpoActive K1 10.0 K2 1.0 GAMMA 35.0 K1COMP 0.0 K2COMP 1.0 INIT 0
       * ADAPT_ANGLE 0 S 54000 LAMBDAM 1.4 LAMBDA0 0.8 DENS 1050
       */
      class CoupAnisoExpoActive : public Mat::PAR::ParameterAniso
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpoActive(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// fiber params
        double k1_;
        double k2_;
        /// angle between circumferential and fiber direction (used for cir, axi, rad nomenclature)
        double gamma_;
        /// fiber params for the compressible case
        double k1comp_;
        double k2comp_;
        /// fiber initialization status
        int init_;
        /// adapt angle during remodeling
        bool adapt_angle_;
        /// maximum contractile stress
        double s_;
        /// stretch at maximum active force generation
        double lambdamax_;
        /// stretch at zero active force generation
        double lambda0_;
        /// total reference mass density at the beginning of the simulation
        double dens_;

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
      };  // class CoupAnisoExpoActive
    }  // namespace PAR

    /*!
     * @brief Coupled anisotropic active fiber function, implemented for one possible fiber family
     * as in [1]
     *
     * This is an active anisotropic material of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = \frac {s}{\rho} \left(1.0 + \frac{1}{3} \frac{\left(\lambda_m - 1.0
     *   \right)^3}{\left(\lambda_m - \lambda_0 \right)^2} \right).
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] Wilson, J.S., S. Baek, and J.D. Humphrey, Parametric study of effects of collagen
     * turnover on the natural history of abdominal aortic aneurysms. Proc. R. Soc. A, 2013.
     * 469(2150): p. 20120556.
     * </ul>
     */
    class CoupAnisoExpoActive : public ActiveSummand, public FiberAnisotropyExtensionProvider<1>
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpoActive(Mat::Elastic::PAR::CoupAnisoExpoActive* params);

      ///@name Packing and Unpacking
      //@{

      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;

      //@}

      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupanisoexpoactive;
      }

      //@}

      /// Setup of active summand
      void setup(int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

      void evaluate_first_derivatives_aniso(Core::LinAlg::Matrix<2, 1>& dPI_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;

      void evaluate_second_derivatives_aniso(Core::LinAlg::Matrix<3, 1>& ddPII_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;

      /// retrieve coefficients of first, second and third derivative
      /// of summand with respect to anisotropic invariants
      /// ATTENTION: this is only the passive contribution of the fiber!
      template <typename T>
      void get_derivatives_aniso(Core::LinAlg::Matrix<2, 1, T>&
                                     dPI_aniso,  ///< first derivative with respect to invariants
          Core::LinAlg::Matrix<3, 1, T>&
              ddPII_aniso,  ///< second derivative with respect to invariants
          Core::LinAlg::Matrix<4, 1, T>&
              dddPIII_aniso,  ///< third derivative with respect to invariants
          Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg,  ///< right Cauchy-Green tensor
          int gp,                                             ///< Gauss point
          int eleGID) const;                                  ///< element GID

      /// Add anisotropic principal stresses
      /// ATTENTION: this is only the passive contribution of the fiber!
      void add_stress_aniso_principal(
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              rcg,  ///< right Cauchy Green in "strain-like" Voigt notation
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,      ///< 2nd PK-stress
          const Teuchos::ParameterList&
              params,  ///< additional parameters for computation of material properties
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// Evaluates strain energy for automatic differentiation with FAD
      template <typename T>
      void evaluate_func(T& psi,                              ///< strain energy functions
          Core::LinAlg::SymmetricTensor<T, 3, 3> const& rcg,  ///< Right Cauchy-Green tensor
          int gp,                                             ///< Gauss point
          int eleGID) const;                                  ///< element GID

      /// evaluate stress and cmat
      /// ATTENTION: this is only the active contribution of the fiber!
      void add_active_stress_cmat_aniso(
          Core::LinAlg::SymmetricTensor<double, 3, 3> const& CM,    ///< rigtht Cauchy Green tensor
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,      ///< 2nd PK-stress
          int gp,                                                   ///< Gauss point
          int eleGID) const override;                               ///< element GID


      /// evaluate stress and cmat
      /// ATTENTION: this is only the active contribution of the fiber!
      template <typename T>
      void evaluate_active_stress_cmat_aniso(
          Core::LinAlg::SymmetricTensor<T, 3, 3> const& CM,    ///< rigtht Cauchy Green tensor
          Core::LinAlg::SymmetricTensor<T, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<T, 3, 3>& stress,      ///< 2nd PK-stress
          int gp,                                              ///< Gauss point
          int eleGID) const;                                   ///< element GID

      // add strain energy
      void add_strain_energy(double& psi,  ///< strain energy functions
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              glstrain,  ///< Green-Lagrange strain in strain like Voigt notation
          int gp,        ///< Gauss point
          int eleGID     ///< element GID
          ) override;

      /// @name Access methods
      //@{
      template <typename T>
      inline void get_derivative_aniso_active(T& dPIact) const
      {
        dPIact = d_p_iact_;
      };

      double get_derivative_aniso_active() const override { return d_p_iact_; };

      //@}

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

      /*!
       * \brief Returns the reference to the used Mat::FiberAnisotropyExtension
       *
       * \return FiberAnisotropyExtension& Reference to the used Mat::AnisotropyExtension
       */
      FiberAnisotropyExtension<1>& get_fiber_anisotropy_extension() override
      {
        return anisotropy_extension_;
      }

     private:
      /// Evaluate the first derivative of the active fiber potential w.r.t active fiber stretch
      double evaluated_psi_active() const;

      /// my material parameters
      Mat::Elastic::PAR::CoupAnisoExpoActive* params_;

      /// first derivative of active fiber potential w.r.t. the active fiber stretch
      double d_p_iact_;

      /// active fiber stretch for a given muscle tone
      double lambdaact_;

      /// Anisotropy extension that manages fibers and structural tensors
      DefaultAnisotropyExtension<1> anisotropy_extension_;
    };  // namespace PAR

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPANISOEXPOBASE_HPP
#define FOUR_C_MAT_ELAST_COUPANISOEXPOBASE_HPP

#include "4C_config.hpp"

#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_mat_anisotropy_extension_base.hpp"
#include "4C_mat_anisotropy_extension_provider.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN


namespace Mat
{
  namespace Elastic
  {
    /*!
     * \brief Pure abstract class to define the interface to the implementation of specific
     * materials
     */
    class CoupAnisoExpoBaseInterface
    {
     public:
      virtual ~CoupAnisoExpoBaseInterface() = default;

      /*!
       * \brief Returns the scalar product of the fibers
       *
       * \param gp Gauss point
       * \return double
       */
      virtual double get_scalar_product(int gp) const = 0;

      /*!
       * \brief Returns the structural tensor that should be used
       *
       * \param gp Gauss point
       * \return const Core::LinAlg::Matrix<3, 3>&
       */
      virtual const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_structural_tensor(
          int gp) const = 0;
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
      class CoupAnisoExpoBase
      {
       public:
        /// standard constructor
        explicit CoupAnisoExpoBase(const Core::Mat::PAR::Parameter::Data& matdata);

        /// Constructor only used for unit testing
        CoupAnisoExpoBase();

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

        //@}
      };  // class CoupAnisoExpoBase

    }  // namespace PAR

    /*!
     * @brief Coupled anisotropic exponential fiber function, implemented for one possible fiber
     * family [1] This is a hyperelastic, anisotropic material of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = \frac {k_1}{2 k_2} \left(e^{k_2 (IV_{\boldsymbol C}-1)^2 }-1 \right).
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] G.A. Holzapfel, T.C. Gasser, R.W. Ogden: A new constitutive framework for arterial
     * wall mechanics and a comparative study of material models, J. of Elasticity 61 (2000) 1-48.
     * </ul>
     */
    class CoupAnisoExpoBase : public Summand
    {
     public:
      /// constructor with given material parameters
      explicit CoupAnisoExpoBase(Mat::Elastic::PAR::CoupAnisoExpoBase* params);

      /*!
       * \brief Evaluate first derivative of the strain energy function with respect to the
       * anisotropic invariants.
       *
       * \param dPI_aniso (out) : First derivatives of the strain energy function with respect to
       * the anisotropic invariants
       * \param C (in) : Cauchy Green deformation tensor
       * \param gp (in) : Gauss point
       * \param eleGID (in) : global element id
       */
      void evaluate_first_derivatives_aniso(Core::LinAlg::Matrix<2, 1>& dPI_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;

      /*!
       * \brief Evaluate second derivative of the strain energy function with respect to the
       * anisotropic invariants.
       *
       * \param ddPI_aniso (out) : Second derivatives of the strain energy function with respect to
       * the anisotropic invariants
       * \param C (in) : Cauchy Green deformation tensor
       * \param gp (in) : Gauss point
       * \param eleGID (in) : global element id
       */
      void evaluate_second_derivatives_aniso(Core::LinAlg::Matrix<3, 1>& ddPII_aniso,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg, int gp, int eleGID) override;


      /*!
       * @brief retrieve coefficients of first, second and third derivative of summand with respect
       * to anisotropic invariants
       *
       * The derivatives of the summand
       * \f$\Psi(IV_{\boldsymbol{C},\boldsymbol{a}},V_{\boldsymbol{C},\boldsymbol{a}})\f$ in which
       * the principal anisotropic invariants are the arguments are defined as following:
       *
       * First derivatives:
       *
       * \f[
       * dPI_{0,aniso} = \frac{\partial \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dPI_{1,aniso} = \frac{\partial \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * Second derivatives:
       * \f[
       * ddPII_{0,aniso} = \frac{\partial^2 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}^2} ;
       * \f]
       * \f[
       * ddPII_{1,aniso} = \frac{\partial^2 \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}^2} ;
       * \f]
       * \f[
       * ddPII_{2,aniso} = \frac{\partial^2 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * Third derivatives:
       * \f[
       * dddPIII_{0,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial IV_{\boldsymbol{C},\boldsymbol{a}} \partial IV_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{1,aniso} = \frac{\partial^3 \Psi}{\partial V_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{2,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial IV_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       * \f[
       * dddPIII_{3,aniso} = \frac{\partial^3 \Psi}{\partial IV_{\boldsymbol{C},\boldsymbol{a}}
       * \partial V_{\boldsymbol{C},\boldsymbol{a}} \partial V_{\boldsymbol{C},\boldsymbol{a}}} ;
       * \f]
       */
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
      void add_stress_aniso_principal(
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,   ///< right Cauchy Green Tensor
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,      ///< 2nd PK-stress
          const Teuchos::ParameterList&
              params,  ///< additional parameters for computation of material properties
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// add strain energy
      void add_strain_energy(double& psi,  ///< strain energy functions
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              glstrain,  ///< Green-Lagrange strain in strain like Voigt notation
          int gp,        //< Gauss point
          int eleGID     ///< element GID
          ) override;

      /// Evaluates strain energy for automatic differentiation with FAD
      template <typename T>
      void evaluate_func(T& psi,                            ///< strain energy functions
          Core::LinAlg::SymmetricTensor<T, 3, 3> const& C,  ///< Right Cauchy-Green tensor
          int gp,                                           ///< Gauss point
          int eleGID) const;                                ///< element GID

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

     protected:
      /*!
       * \brief Get the structural tensor interface from the derived materials
       *
       * \return const CoupAnisoExpoBaseInterface& Interface that computes structural tensors and
       * scalar products
       */
      virtual const CoupAnisoExpoBaseInterface& get_coup_aniso_expo_base_interface() const = 0;

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupAnisoExpoBase* params_;
    };  // namespace Elastic

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

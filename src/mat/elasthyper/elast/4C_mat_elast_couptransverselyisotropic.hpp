// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPTRANSVERSELYISOTROPIC_HPP
#define FOUR_C_MAT_ELAST_COUPTRANSVERSELYISOTROPIC_HPP

#include "4C_config.hpp"

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
       * \brief parameter container for the simple orthotropic, transversely
       * isotropic hyperelastic constitutive equation for large strain computations
       *
       * These parameters belong to a simple orthotropic, transversely isotropic
       * material model which is suitable for large strains. The main purpose is to
       * combine it with a simple hyperelastic material model (e.g. St. Venant Kirchhoff or
       * Neo-Hookean), such that a easy parameter identification becomes possible
       * by consideration of the parameters for the small strain regime via
       *
       * \f{eqnarray}{
       * \lambda &=& \Lambda = \frac{\nu_{12} \nu_{21} + \nu_{22}}{(1-\nu_{22}-2\nu_{12}
       * \nu_{21})(1+\nu_{22})} E_{2}, \\
       * \mu &=& G_{23} = \frac{E_{2}}{2(1+\nu_{23})}, \\
       * \alpha &=& G_{23} - G_{12}, \\
       * \beta &=& \nu_{12} (\lambda + G_{23}) - \frac{\lambda}{2}, \\
       * \gamma &=& \frac{1}{8} \left(\frac{1 -\nu_{23}}{1-\nu_{23} - \nu_{12} \nu_{21}} E_{1} + 4
       * \alpha - 4 \beta - \lambda - 2 \mu \right). \f} where \f$ E_{2} = E_{3} \f$ is the Young
       * modulus in the plane normal to the fiber direction and \f$ E_{1} = E_{A} \f$ denotes the
       * Young's modulus in fiber direction. \f$ \nu_{12} \f$ is the Poisson ratio for tension in
       * fiber direction. Furthermore, \f$ G_{12}=G_{13}=G_{A} \f$ represents shear modulus in fiber
       * direction, while the shear modulus in the normal plane \f$ G_{23} \f$ dependents on the
       * already known isotropic parameters via \f$ G_{23} = (E_{2}) / (2(1+\nu_{23})) \f$.
       * Furthermore, the relation \f$ \frac{\nu_{21}}{E_{2}} = \frac{\nu_{12}}{E_{1}} \f$
       * holds.
       *
       * The implementation follows partly the reference
       *
       * J. Bonet, A. J. Burton, "A simple orthotropic, transversely isotropic
       * hyperelastic constitutive equation for large strain computations",
       * Comput. Methods Appl. Mech. Engrg., 162, pp. 151-164, (1998)
       *
       * The parameter identification is based on the linear compliance matrix
       * given in
       *
       * H. Schuermann, "Konstruieren mit Faser-Kunststoff-Verbunden", Springer, 2005.
       */
      class CoupTransverselyIsotropic : public Mat::PAR::ParameterAniso
      {
       public:
        /// no default constructor
        CoupTransverselyIsotropic() = delete;

        /// standard constructor
        explicit CoupTransverselyIsotropic(const Core::Mat::PAR::Parameter::Data& matdata);

        /// alpha parameter of the constitutive law
        double alpha_ = 0.0;

        /// beta parameter of the constitutive law
        double beta_ = 0.0;

        /// gamma parameter of the constitutive law
        double gamma_ = 0.0;

        /// angle of the fiber direction
        double angle_ = 0.0;

        /// global fiber id number (1,2,3,...) used later as FIBER1,FIBER2,FIBER3,...
        int fiber_gid_ = 0;

        /// fiber initialization status
        int init_ = 0;

        /// print parameters
        void print() const;

       private:
        /// Override this method and throw error, as the material should be created in within the
        /// Factory method of the elastic summand
        std::shared_ptr<Core::Mat::Material> create_material() override
        {
          FOUR_C_THROW(
              "Cannot create a material from this method, as it should be created in "
              "Mat::Elastic::Summand::Factory.");
          return nullptr;
        };
      };  // class CoupAnisoSimple
    }  // namespace PAR

    /*!
     * \brief Implementation of the simple orthotropic, transversely isotropic hyperelastic
     * constitutive equation for large strain computations
     *
     *  The implementation follows partly the reference
     *
     *  J. Bonet, A. J. Burton, "A simple orthotropic, transversely isotropic
     *  hyperelastic constitutive equation for large strain computations",
     *  Comput. Methods Appl. Mech. Engrg., 162, pp. 151-164, (1998).
     *
     *  \note The given reference contains many typos, such that almost all
     *  equations (beside the strain energy function) were derived independently.
     */
    class CoupTransverselyIsotropic : public Summand
    {
      using my_params = Mat::Elastic::PAR::CoupTransverselyIsotropic;

     public:
      /// no default constructor
      CoupTransverselyIsotropic() = delete;

      /// standard constructor
      explicit CoupTransverselyIsotropic(my_params* params);

      /// @name Packing and Unpacking
      /// @{

      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;

      /// @}

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_couptransverselyisotropic;
      }

      /// Setup of summand
      void setup(int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      /*!
       * \brief add strain energy [derived]
       *
       * \f[
       * \Psi_{\mathrm{trn}} = \left( \alpha + \frac{\beta}{2} \ln(I_{3}) +
       * \gamma (I_{4}-1) \right) (I_{4}-1) - \frac{\alpha}{2} ( I_{5} - 1)
       * \f]
       *
       * \note \f$ \ln(J) = \ln(\sqrt(I_{3})) = \frac{1}{2} \ln(I_{3}) \f$
       * holds.
       *
       * \param[out] psi      Updated value of the strain energy function
       * \param[in]  prinv    principal invariants of right Cauchy-Green tensor
       * \param[in]  modinv   modified invariants of right Cauchy-Green tensor
       * \param[in]  glstrain Green-Lagrange strain in strain like Voigt notation
       * \param[in]  go       Gauss point
       * \param[in]  eleGID   element global ID
       */
      void add_strain_energy(double& psi, const Core::LinAlg::Matrix<3, 1>& prinv,
          const Core::LinAlg::Matrix<3, 1>& modinv,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const int gp,
          const int eleGID) override;

      /*!
       * \brief Add anisotropic principal stresses
       *
       * \param[in]  rcg    right Cauchy Green Tensor
       * \param[out] cmat   material stiffness matrix
       * \param[out] stress 2nd PK-stress
       * \param[in]  params additional parameters for comp. of material properties
       * \param[in]  go       Gauss point
       * \param[in]  eleGID element GID
       */
      void add_stress_aniso_principal(const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress, const Teuchos::ParameterList& params,
          const int gp, const int eleGID) override;

      /*!
       * \brief Set fiber directions
       *
       * \param[in] newangle  new angle for the fiber direction
       * \param[in] locsys    local coordinate system
       * \param[in] defgrd    deformation gradient
       */
      void set_fiber_vecs(const double newangle, const Core::LinAlg::Tensor<double, 3, 3>& locsys,
          const Core::LinAlg::Tensor<double, 3, 3>& defgrd) override;

      /// Get fiber directions
      void get_fiber_vecs(
          std::vector<Core::LinAlg::Tensor<double, 3>>& fibervecs  ///< vector of all fiber vectors
      ) const override;

      /*!
       * \brief Indicator for formulation
       *
       * \param[out] isoprinc     global indicator for isotropic principal formulation
       * \param[out] isomod       global indicator for isotropic split formulation
       * \param[out] anisoprinc   global indicator for anisotropic principal formulation
       * \param[out] anisomod     global indicator for anisotropic split formulation
       * \param[out] viscogeneral global indicator, if one viscoelastic formulation is used
       */
      void specify_formulation(bool& isoprinc, bool& isomod, bool& anisoprinc, bool& anisomod,
          bool& viscogeneral) override
      {
        anisoprinc = true;
        return;
      };

     private:
      /*!
       * \brief Reset the pseudo invariants and the jacobian determinant
       *
       * \param[in] rcg right cauchy green tensor in perturbed Voigt strain notation
       */
      void reset_invariants(const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg);

      /*!
       * \brief Add the material contributions to the second Piola Kirchhoff stress tensor
       *
       * \f{eqnarray}{
       * \mathbf{S}_{\mathrm{trn}} =& 2 \left\{ \frac{\partial \Psi_{\mathrm{trn}} }{\partial
       * I_{3}} \frac{\partial I_{3}}{\partial \mathbf{C}}
       * + \frac{\partial \Psi_{\mathrm{trn}} }{\partial I_{4}} \frac{\partial I_{4}}{\partial
       * \mathbf{C}}
       * + \frac{\partial \Psi_{\mathrm{trn}} }{\partial I_{5}} \frac{\partial I_{5}}{\partial
       * \mathbf{C}} \right\} \nonumber \\
       * =& 2 \left\{
       * \frac{\beta}{2 I_{3}} (I_{4}-1) I_{3} \mathbf{C}^{-1} + \left[\alpha + \frac{\beta}{2}
       * \ln(I_{3}) + 2 \gamma (I_{4}-1)\right] \mathbf{A} \otimes \mathbf{A}
       * - \frac{\alpha}{2} ( \mathbf{A} \otimes \mathbf{C}\, \mathbf{A} + \mathbf{A}\, \mathbf{C}
       * \otimes \mathbf{A}
       * )\right\} \nonumber \\
       * =& \beta (I_{4} - 1) \mathbf{C}^{-1} + 2\left[\alpha + \frac{\beta}{2} \ln(I_{3}) + 2
       * \gamma (I_{4}-1)\right] \mathbf{A} \otimes \mathbf{A}
       * - \alpha ( \mathbf{A} \otimes \mathbf{C}\, \mathbf{A} + \mathbf{A}\, \mathbf{C} \otimes
       * \mathbf{A} ). \f}
       *
       * \param[out] stress     Updated second Piola Kirchhoff stress tensor
       * \param[in]  rcg_s      right Cauchy Green strain tensor in perturbed
       *                        Voigt stress notation
       * \param[out] rcg_inv_s  Inverse of the right Cauchy green strain tensor
       *                        in perturbed Voigt stress notation
       */
      void update_second_piola_kirchhoff_stress(Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_s,
          Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_inv_s) const;

      /*!
       * \brief Update the elasticity tensor
       *
       * \f{eqnarray}{
       * \mathbb{C}_{\mathrm{trn}} = [\mathbb{C}_{\mathrm{trn}}]_{ABCD} =& 2 \frac{\partial
       * [\mathbf{S}_{\mathrm{trn}}]_{AB}}{\partial [\mathbf{C}]_{CD}} =
       * 2 \{ \beta [\mathbf{C}^{-1}]_{AB} [\underline{A} \otimes \underline{A}]_{CD}  \nonumber \\
       * & - \beta (I_{4} -1 ) \frac{1}{2} \left([\mathbf{C}^{-1}]_{AC}[\mathbf{C}^{-1}]_{BD} +
       * [\mathbf{C}^{-1}]_{AD} [\mathbf{C}^{-1}]_{BC} \right) \nonumber \\
       * &+ \beta \frac{1}{I_{3}} I_{3} [\mathbf{C}^{-1}]_{CD} [\underline{A} \otimes
       * \underline{A}]_{AB} + 4
       * \gamma  [\underline{A} \otimes \underline{A} \otimes \underline{A} \otimes
       * \underline{A}]_{ABCD} \nonumber \\
       * &-\frac{\alpha}{2} \left( [\underline{A}]_{A} [\underline{A}]_{D} [\mathbf{\delta}]_{BC} +
       * [\underline{A}]_{A} [\underline{A}]_{C} [\mathbf{\delta}]_{BD} +
       * [\underline{A}]_{C} [\underline{A}]_{B} [\mathbf{\delta}]_{AD} \right. \nonumber \\
       * &\left. +
       * [\underline{A}]_{D} [\underline{A}]_{B} [\mathbf{\delta}]_{AC} \right) \nonumber \\
       * =& 2 \beta \left( \mathbf{C}^{-1} \otimes \underline{A} \otimes \underline{A} +
       * \underline{A} \otimes \underline{A}\otimes \mathbf{C}^{-1} \right) -2 \beta (I_{4}-1)
       * \mathbf{C}^{-1} \odot \mathbf{C}^{-1}
       * \nonumber \\
       * &-2 \alpha \left( (\underline{A} \otimes \underline{A} ) \odot \mathbf{\delta}
       * + \mathbf{\delta} \odot (\underline{A} \otimes \underline{A} ) \right)
       * \f}
       *
       * The \f$\odot\f$ symbol follows the definition given in Holzapfel's book
       * "Nonlinear Solid Mechanics - A continuum approach for engineering" on
       * page 254, equation (6.164) and (6.165).
       *
       * \param[out] cmat       updated elasticity tensor
       * \param[in]  rcg_inv_s  right cauchy green tensor in perturbed Voigt
       *                        stress notation
       */
      void update_elasticity_tensor(Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg_inv_s) const;

      /// pointer to the fiber parameters
      my_params* params_ = nullptr;

      /// fiber direction
      Core::LinAlg::Tensor<double, 3> a_;

      /** \brief outer product of the fiber directions
       * \f$ \underline{A} \otimes \underline{A}\f$
       *
       * \note We are following a perturbed Voigt notation:
       * {11, 22, 33, 12, 23, 13}. */
      Core::LinAlg::SymmetricTensor<double, 3, 3> aa_;

      /// pseudo invariant \f$ I_4 \f$ ( strain measure in fiber direction )
      double i4_ = 0.0;

      /// pseudo invariant \f$ I_5 \f$ ( quadratic strain measure in fiber direction )
      double i5_ = 0.0;

      /// determinant of the deformation gradient
      double j_ = 0.0;
    };  // class CoupAnisoSimple

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

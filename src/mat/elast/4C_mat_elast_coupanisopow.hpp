// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_COUPANISOPOW_HPP
#define FOUR_C_MAT_ELAST_COUPANISOPOW_HPP

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
       * @brief material parameters for anisochoric contribution of a pow like material with one
       * fiber direction
       *
       * <h3>Input line</h3>
       * MAT 1 CoupAnisoPow C 1.0 D 2.0 [ GAMMA 35.0 INIT 0 ADAPT_ANGLE 0]
       */
      class CoupAnisoPow : public Mat::PAR::ParameterAniso
      {
       public:
        /// standard constructor
        CoupAnisoPow(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// stiffness factor
        double k_;
        /// exponential factor for I4
        double d1_;
        /// exponential factor for ((I4)^{d1}-1)
        double d2_;
        /// fiber number (1,2,3,...) used later as FIBER1,FIBER2,FIBER3,...
        int fibernumber_;
        /// Deformation threshold where fibers are active.
        double activethres_;
        /// angle between circumferential and fiber direction (used for cir, axi, rad nomenclature)
        double gamma_;
        /// fiber initialization status
        int init_;
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
      };  // class CoupAnisoPow

    }  // namespace PAR

    /*!
     * @brief Coupled anisotropic pow-like fiber function, implemented for one possible fiber family
     * as in [1]
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = K \left((IV_{\boldsymbol C})^{D1}-1\right)^{D2}.
     * \f]
     *
     * The corresponding derivatives are
     * \f[
     *   \frac{d\Psi}{d IV_{\boldsymbol C}} = K\ D2\ D1\ (IV_{\boldsymbol C})^{D1-1}
     *   \left((IV_{\boldsymbol C})^{D1}-1\right)^{D2-1}.
     * \f]
     * and
     * \f[
     *   \frac{d^2\Psi}{d^2 IV_{\boldsymbol C}} = K\ D2\ (D2-1)\ \left( D1\ (IV_{\boldsymbol
     *   C})^{D1-1} \right)^2 \left((IV_{\boldsymbol C})^{D1}-1\right)^{D2-2}\ +\ K\ D2\ D1
     *   (D1-1)\ (IV_{\boldsymbol C})^{D1-2}\ \left((IV_{\boldsymbol C})^{D1}-1\right)^{D2-1}
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] GA Holzapfel, Nonlinear solid mechanics 2004
     * </ul>
     */
    class CoupAnisoPow : public Summand
    {
     public:
      /// constructor with given material parameters
      CoupAnisoPow(Mat::Elastic::PAR::CoupAnisoPow* params);

      ///@name Packing and Unpacking
      //@{
      void pack_summand(Core::Communication::PackBuffer& data) const override;

      void unpack_summand(Core::Communication::UnpackBuffer& buffer) override;
      //@}

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_coupanisoneohooke;
      }

      /// Setup of summand
      void setup(int numgp, const Core::IO::InputParameterContainer& container) override;

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
      void set_fiber_vecs(const double newgamma,             ///< new angle
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
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::CoupAnisoPow* params_;

      /// fiber direction
      Core::LinAlg::Tensor<double, 3> a_;

      /// structural tensors in Voigt notation for anisotropy
      Core::LinAlg::SymmetricTensor<double, 3, 3> structural_tensor_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

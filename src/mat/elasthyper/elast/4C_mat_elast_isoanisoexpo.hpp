// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISOANISOEXPO_HPP
#define FOUR_C_MAT_ELAST_ISOANISOEXPO_HPP

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
       * @brief material parameters for isochoric contribution of a anisotropic exponential fiber
       * material
       *
       * <h3>Input line</h3>
       * MAT 1 ELAST_IsoAnisoExpo K1 10.0 K2 1.0 GAMMA 35.0 K1COMP 0.0 K2COMP 1.0 INIT 0 ADAPT_ANGLE
       * 0
       */
      class IsoAnisoExpo : public Mat::PAR::ParameterAniso
      {
       public:
        /// standard constructor
        IsoAnisoExpo(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// fiber params
        double k1_;
        double k2_;
        /// angle between circumferential and fiber direction
        double gamma_;
        /// fiber params for the compressible case
        double k1comp_;
        double k2comp_;
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
      };  // class IsoAnisoExpo

    }  // namespace PAR

    /*!
     * @brief Isochoric anisotropic exponential fiber function, implemented for one possible fiber
     * family [1]
     *
     * This is a hyperelastic, anisotropic material of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *    \Psi = \frac {k_1}{2 k_2} \left(e^{k_2 (\overline{IV}_{\boldsymbol C}-1)^2 }-1 \right).
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] G.A. Holzapfel, T.C. Gasser, R.W. Ogden: A new constitutive framework for arterial
     * wall mechanics and a comparative study of material models, J. of Elasticity 61 (2000) 1-48.
     * </ul>
     */
    class IsoAnisoExpo : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoAnisoExpo(Mat::Elastic::PAR::IsoAnisoExpo* params);

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
        return Core::Materials::mes_isoanisoexpo;
      }

      //@}

      /// Setup of summand
      void setup(int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      /// Add anisotropic modified stresses
      void add_stress_aniso_modified(
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& rcg,  ///< right Cauchy Green Tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>&
              icg,  ///< inverse of right Cauchy Green Tensor
          Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat,  ///< material stiffness matrix
          Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,      ///< 2nd PK-stress
          double I3,                                                ///< third principal invariant
          int gp,                                                   ///< Gauss point
          int eleGID,                                               ///< element GID
          const EvaluationContext<3>& context  ///< Container for additional information
          ) override;

      /// retrieve coefficients of first, second and third derivative
      /// of summand with respect to anisotropic invariants
      virtual void get_derivatives_aniso(
          Core::LinAlg::Matrix<2, 1>& dPI_aniso,  ///< first derivative with respect to invariants
          Core::LinAlg::Matrix<3, 1>&
              ddPII_aniso,  ///< second derivative with respect to invariants
          Core::LinAlg::Matrix<4, 1>&
              dddPIII_aniso,  ///< third derivative with respect to invariants
          double I4,          ///< fourth invariant
          int gp,             ///< Gauss point
          int eleGID);        ///< element GID

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
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
        anisomod = true;
      };

     protected:
      /// my material parameters
      Mat::Elastic::PAR::IsoAnisoExpo* params_;

      /// fiber direction
      Core::LinAlg::Tensor<double, 3> a_;
      /// structural tensors in voigt notation for anisotropy
      Core::LinAlg::SymmetricTensor<double, 3, 3> structural_tensor_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ISONEOHOOKE_HPP
#define FOUR_C_MAT_ELAST_ISONEOHOOKE_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
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
       * @brief material parameters for isochoric contribution of a neo-Hooke material
       */
      class IsoNeoHooke : public Core::Mat::PAR::Parameter
      {
       public:
        /// standard constructor
        IsoNeoHooke(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// Shear modulus
        Core::IO::InputField<double> mue_;

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
      };  // class IsoNeoHooke
    }  // namespace PAR

    /*!
     * @brief Isochoric neo-Hooke material according to [1]
     *
     * This is a hyperelastic, isotropic material
     * of the most simple kind.
     *
     * Strain energy function is given by
     * \f[
     *   \Psi = \frac{\mu}{2} (\overline{I}_{\boldsymbol{C}}-3) = \frac{\mu}{2}
     *   (J^{-2/3}{I}_{\boldsymbol{C}}-3)
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] GA Holzapfel, "Nonlinear solid mechanics", Wiley, 2000.
     * </ul>
     */
    class IsoNeoHooke : public Summand
    {
     public:
      /// constructor with given material parameters
      IsoNeoHooke(Mat::Elastic::PAR::IsoNeoHooke* params);


      /// @name Access material constants
      //@{

      /// material type
      Core::Materials::MaterialType material_type() const override
      {
        return Core::Materials::mes_isoneohooke;
      };

      /// add shear modulus equivalent
      void add_shear_mod(bool& haveshearmod,  ///< non-zero shear modulus was added
          double& shearmod,                   ///< variable to add upon
          int ele_gid                         ///< element GID
      ) const override
      {
        haveshearmod = true;
        shearmod += params_->mue_.at(ele_gid);
      };

      //@}

      // add strain energy
      void add_strain_energy(double& psi,  ///< strain energy function
          const Core::LinAlg::Matrix<3, 1>&
              prinv,  ///< principal invariants of right Cauchy-Green tensor
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,  ///< Green-Lagrange strain
          int gp,                                                       ///< Gauss point
          int eleGID                                                    ///< element GID
          ) override;

      // Add derivatives with respect to modified invariants.
      void add_derivatives_modified(
          Core::LinAlg::Matrix<3, 1>&
              dPmodI,  ///< first derivative with respect to modified invariants
          Core::LinAlg::Matrix<6, 1>&
              ddPmodII,  ///< second derivative with respect to modified invariants
          const Core::LinAlg::Matrix<3, 1>&
              modinv,  ///< modified invariants of right Cauchy-Green tensor
          int gp,      ///< Gauss point
          int eleGID   ///< element GID
          ) override;

      /// @name Access methods
      //@{
      double mue(int ele_gid) const { return params_->mue_.at(ele_gid); }
      //@}

      /// Indicator for formulation
      void specify_formulation(
          bool& isoprinc,     ///< global indicator for isotropic principal formulation
          bool& isomod,       ///< global indicator for isotropic split formulation
          bool& anisoprinc,   ///< global indicator for anisotropic principal formulation
          bool& anisomod,     ///< global indicator for anisotropic split formulation
          bool& viscogeneral  ///< general indicator, if one viscoelastic formulation is used
          ) override
      {
        isomod = true;
        return;
      };

     private:
      /// my material parameters
      Mat::Elastic::PAR::IsoNeoHooke* params_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_ELAST_ANISOACTIVESTRESS_EVOLUTION_HPP
#define FOUR_C_MAT_ELAST_ANISOACTIVESTRESS_EVOLUTION_HPP

#include "4C_config.hpp"

#include "4C_mat_anisotropy.hpp"
#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_elast_summand.hpp"
#include "4C_mat_par_aniso.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_solid_ele_fibers.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace Elastic
  {
    namespace PAR
    {
      /*!
       * <h3>Input line</h3>
       * MAT 1 ELAST_AnisoActiveStress_Evolution SIGMA 100.0 TAUC0 0.0 MAX_ACTIVATION 30.0
       * MIN_ACTIVATION -20.0 SOURCE_ACTIVATION 1 ACTIVATION_THRES 0 [STRAIN_DEPENDENCY No]
       * [LAMBDA_LOWER 0.707] [LAMBDA_UPPER 1.414]
       */
      class AnisoActiveStressEvolution : public Mat::PAR::ParameterAniso
      {
       public:
        /// standard constructor
        explicit AnisoActiveStressEvolution(const Core::Mat::PAR::Parameter::Data& matdata);

        /// @name material parameters
        //@{

        /// fiber params
        double sigma_;
        /// initial condition
        double tauc0_;
        /// Maximal value for rescaling the activation curve
        double maxactiv_;
        /// Minimal value for rescaling the activation curve
        double minactiv_;
        /// Threshold for stress activation function
        double activationthreshold_;
        /// Where the activation comes from: 0=scatra , >0 Id for FUNCT
        int sourceactiv_;
        /// is there a strain dependency for the active tension?
        bool strain_dep_;
        /// lower stretch threshold
        double lambda_lower_;
        /// upper stretch threshold
        double lambda_upper_;
        /// fiber angle
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
      };  // class AnisoActiveStress_Evolution

    }  // namespace PAR

    /*!
     * This is a simplification of the muscle contraction law proposed in [1],[2],
     * resulting in the following first order ODE for the active stress tau, compare [3]:
     *
     * \f[
     *   \frac{d}{dt} \tau = n_0 \sigma_0 |u|_+ - \tau |u|, \quad \tau(0) = tau_0
     * \f]
     *
     * where \f$\sigma_0\f$ is the contractility (asymptotic value of \tau) and u is a control
     * variable either provided by a electrophysiology simulation or by a user-specified function
     * therein, n0 is a strain-dependent factor that may take into account the Frank-Starling
     * effect! n0 \in [0; 1] scales the contractility depending on a lower and upper fiber stretch
     * lambda using a flipped parabola, n0 = -(lambda - lambda_lower)*(lambda - lambda_upper) *
     * 4/(lambda_lower-lambda_upper)^2, which is an approximation to the function from Sainte-Marie
     * et al. 2006, Fig. 2(ii) other laws might be thought of here, since hardly any literature
     * provides a meaningful dependency...
     *
     * Due to the active stress approach, see [4], the active stress will be added along a given
     * fiber direction f_0 to the 2nd Piola-Kirchhoff stress:
     * \f[
     *   S_{active} = \tau(t) f_0 \otimes f_0
     * \f]
     *
     * <h3>References</h3>
     * <ul>
     * <li> [1] 2012 Chapelle, Le Tallec, Moireau, Sorine - An energy-preserving muscle tissue
     * model: formulation and compatible discretizations, Journal for Multiscale Computational
     * Engineering 10(2):189-211 (2012) <li> [2] 2001 Bestel, Clement, Sorine - A Biomechanical
     * Model of Muscle Contraction (2001), Medical Image Computing and Computer-Assisted
     * Intervention (MICCAI'01), vol. 2208, Springer-Verlag Berlin, 1159-1161 <li> [3] 2002
     * Sermesant, Coudier, Delingette, Ayache - Progress towards an electromechanical model of the
     * heart for cardiac image analysis. (2002) IEEE International Symposium on Biomedical Imaging,
     * 10-13 <li> [4] 1998 Hunter, McCulloch, ter Keurs - Modelling the mechanical properties of
     * cardiac muscle (1998), Progress in Biophysics and Molecular Biology
     * </ul>
     */
    class AnisoActiveStressEvolution : public Summand
    {
     public:
      /// constructor with given material parameters
      explicit AnisoActiveStressEvolution(Mat::Elastic::PAR::AnisoActiveStressEvolution* params);

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
        return Core::Materials::mes_anisoactivestress_evolution;
      }

      //@}

      /*!
       * \brief Register the internally used AnisotropyExtension
       *
       * \param anisotropy Reference to the anisotropy manager
       */
      void register_anisotropy_extensions(Mat::Anisotropy& anisotropy) override;

      /// Setup of summand
      void setup(int numgp, const Discret::Elements::Fibers& fibers,
          const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

      /*!
       * \brief post_setup routine of the element
       *
       * Here potential nodal fibers were passed to the Anisotropy framework
       *
       * @param params Container that potentially contains nodal fibers
       */
      void post_setup(const Teuchos::ParameterList& params) override;

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

      // update internal stress variables
      void update() override;

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
      Mat::Elastic::PAR::AnisoActiveStressEvolution* params_;

      /// Active stress at current time step
      double tauc_np_;
      /// Active stress at last time step
      double tauc_n_;

      /// Special anisotropic behavior
      DefaultAnisotropyExtension<1> anisotropy_extension_;
    };

  }  // namespace Elastic
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

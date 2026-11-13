// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_OGDEN_TCA_HPP
#define FOUR_C_MAT_OGDEN_TCA_HPP

#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

#include <memory>
#include <optional>

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    class OgdenTCA : public Core::Mat::PAR::Parameter
    {
     public:
      /// Constructor
      OgdenTCA(const Core::Mat::PAR::Parameter::Data& matdata);

      std::shared_ptr<Core::Mat::Material> create_material() override;

      /// @name material parameters
      //@{
      const double c_;        ///< stiffness scaling parameter
      const double m_;        ///< nonlinearity parameter
      const double q_;        ///< tension-compression asymmetry control parameter
      const double kappa_;    ///< incompressibility parameter
      const double density_;  ///< density
      //! @}
    };  // end class OgdenTCA
  }  // end namespace PAR


  class OgdenTCAType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "OgdenTCAType"; }

    static OgdenTCAType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static OgdenTCAType instance_;
  };

  /*!
   * \brief Ogden hyperelastic material model with control of tension-compression asymmetry
   *
   * The Ogden TCA model is described in [1]. For a classical hyperelastic Ogden material model,
   * parametrized in terms of the stiffness scaling factor c and the nonlinearity exponent m, the
   * degree of asymmetry cannot be controlled independently from the material nonlinearity. A hybrid
   * formulation is therefore proposed, which enables dedicated control of tension-compression
   * asymmetry through the parameter q in [0, 1]:
   * - q = 0.5: tension-compression symmetric formulation
   *   i.e. the Cauchy stresses |sigma(lambda)| = |sigma(1/lambda)|
   * - q > 0.5: material is stiffer in tension than in compression
   *   i.e. |sigma(lambda)| > |sigma(1/lambda)| for lambda > 1
   * - q < 0.5: material is stiffer in compression than in tension
   *   i.e. |sigma(lambda)| < |sigma(1/lambda)| for lambda > 1
   *
   * Nearly incompressible behavior is enforced using the coupled approach presented in [1]. The
   * parameter kappa controls incompressibility, where higher values of kappa correspond to a more
   * incompressible response.
   *
   * The second Piola--Kirchhoff stress is computed as
   * \f[ \mathbf{S} = \sum_{i=1}^{3}
   * \frac{c}{m} (q \lambda_i^{m-2} - (1-q) \lambda_i^{-m-2}) \mathbf{N}_i \otimes \mathbf{N}_i
   * + [\kappa J(J-1) + \frac{c}{m}(1-2q)] \mathbf{C}^{-1} \f]
   *
   * where \f$J=\det\mathbf{F}=\sqrt{\det\mathbf{C}}\f$,
   * \f$\lambda_i\f$ are the principal stretches, and
   * \f$\mathbf{N}_i\f$ are the corresponding principal directions of \f$\mathbf{C}\f$.
   *
   * Reference:
   * [1] Kevin M. Moerman, Ciaran K. Simms, Thomas Nagel, "Control of tension-compression asymmetry
   * in Ogden hyperelasticity with application to soft tissue modelling", Journal of the Mechanical
   * Behavior of Biomedical Materials, Vol. 56, 2016, pp. 218-228.
   * https://doi.org/10.1016/j.jmbbm.2015.11.027
   */
  class OgdenTCA : public So3Material
  {
   public:
    // Constructor for an empty material
    OgdenTCA();

    // Constructor for a material given the material parameters
    explicit OgdenTCA(Mat::PAR::OgdenTCA* params);

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<OgdenTCA>(*this);
    }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_ogden_tca;
    };

    void valid_kinematics(Solid::KinemType kinem) override
    {
      if (kinem != Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    [[nodiscard]] double density() const override { return params_->density_; }

    [[nodiscard]] int unique_par_object_id() const override
    {
      return OgdenTCAType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    void setup(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void update(const Core::LinAlg::Tensor<double, 3, 3>& defgrd, int const gp,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        int const eleGID) override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

   private:
    /// Material parameters
    Mat::PAR::OgdenTCA* params_{};
  };  // end class OgdenTCA

}  // end namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif
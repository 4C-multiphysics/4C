// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MAT_STVENANTKIRCHHOFF_HPP
#define FOUR_C_MAT_STVENANTKIRCHHOFF_HPP


#include "4C_config.hpp"

#include "4C_comm_parobjectfactory.hpp"
#include "4C_linalg_fixedsizematrix.hpp"
#include "4C_linalg_symmetric_tensor.hpp"
#include "4C_linalg_tensor_conversion.hpp"
#include "4C_linalg_tensor_generators.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"

FOUR_C_NAMESPACE_OPEN

namespace Mat
{
  namespace PAR
  {
    /*----------------------------------------------------------------------*/
    /// material parameters for St. Venant--Kirchhoff
    class StVenantKirchhoff : public Core::Mat::PAR::Parameter
    {
     public:
      /// standard constructor
      StVenantKirchhoff(const Core::Mat::PAR::Parameter::Data& matdata);

      /// @name material parameters
      //@{

      /// Young's modulus
      const double youngs_;
      /// Possion's ratio
      const double poissonratio_;
      /// mass density
      const double density_;

      //@}

      std::shared_ptr<Core::Mat::Material> create_material() override;

    };  // class StVenantKirchhoff
  }  // namespace PAR

  class StVenantKirchhoffType : public Core::Communication::ParObjectType
  {
   public:
    [[nodiscard]] std::string name() const override { return "StVenantKirchhoffType"; }

    static StVenantKirchhoffType& instance() { return instance_; };

    Core::Communication::ParObject* create(Core::Communication::UnpackBuffer& buffer) override;

   private:
    static StVenantKirchhoffType instance_;
  };

  /*----------------------------------------------------------------------*/
  /// Wrapper for St.-Venant-Kirchhoff material
  class StVenantKirchhoff : public So3Material
  {
   public:
    /// construct empty material object
    StVenantKirchhoff();

    /// construct the material object given material parameters
    explicit StVenantKirchhoff(Mat::PAR::StVenantKirchhoff* params);

    [[nodiscard]] int unique_par_object_id() const override
    {
      return StVenantKirchhoffType::instance().unique_par_object_id();
    }

    void pack(Core::Communication::PackBuffer& data) const override;

    void unpack(Core::Communication::UnpackBuffer& buffer) override;

    //@}

    //! @name Access methods

    [[nodiscard]] Core::Materials::MaterialType material_type() const override
    {
      return Core::Materials::m_stvenant;
    }

    void valid_kinematics(Inpar::Solid::KinemType kinem) override
    {
      if (kinem != Inpar::Solid::KinemType::linear &&
          kinem != Inpar::Solid::KinemType::nonlinearTotLag)
        FOUR_C_THROW("element and material kinematics are not compatible");
    }

    [[nodiscard]] std::shared_ptr<Core::Mat::Material> clone() const override
    {
      return std::make_shared<StVenantKirchhoff>(*this);
    }

    /// Young's modulus
    [[nodiscard]] double youngs() const { return params_->youngs_; }

    /// Poisson's ratio
    [[nodiscard]] double poisson_ratio() const { return params_->poissonratio_; }

    [[nodiscard]] double density() const override { return params_->density_; }

    /// shear modulus
    [[nodiscard]] double shear_mod() const
    {
      return 0.5 * params_->youngs_ / (1.0 + params_->poissonratio_);
    }

    [[nodiscard]] Core::Mat::PAR::Parameter* parameter() const override { return params_; }

    //@}

    //! @name Evaluation methods

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>* defgrad,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const Teuchos::ParameterList& params, const EvaluationContext& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double strain_energy(const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain,
        const EvaluationContext& context, int gp, int eleGID) const override;
    //@}

    static constexpr Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_stress(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& glstrain, const double E,
        const double nu)
    {
      const double mue = E / (2 * (1 + nu));
      const double lambda = E * nu / ((1 + nu) * (1 - 2 * nu));
      return lambda * Core::LinAlg::trace(glstrain) *
                 Core::LinAlg::TensorGenerators::identity<double, 3, 3> +
             2 * mue * glstrain;
    }

    static Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> evaluate_stress_linearization(
        const double E, const double nu)
    {
      Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> cmat{};
      Core::LinAlg::Matrix<6, 6> cmat_view = Core::LinAlg::make_stress_like_voigt_view(cmat);

      const double mfac = E / ((1.0 + nu) * (1.0 - 2.0 * nu));  // factor
      // write non-zero components
      cmat_view(0, 0) = mfac * (1.0 - nu);
      cmat_view(0, 1) = mfac * nu;
      cmat_view(0, 2) = mfac * nu;
      cmat_view(1, 0) = mfac * nu;
      cmat_view(1, 1) = mfac * (1.0 - nu);
      cmat_view(1, 2) = mfac * nu;
      cmat_view(2, 0) = mfac * nu;
      cmat_view(2, 1) = mfac * nu;
      cmat_view(2, 2) = mfac * (1.0 - nu);
      // ~~~
      cmat_view(3, 3) = mfac * 0.5 * (1.0 - 2.0 * nu);
      cmat_view(4, 4) = mfac * 0.5 * (1.0 - 2.0 * nu);
      cmat_view(5, 5) = mfac * 0.5 * (1.0 - 2.0 * nu);

      return cmat;
    }


   private:
    /// my material parameters
    Mat::PAR::StVenantKirchhoff* params_;
  };
}  // namespace Mat

FOUR_C_NAMESPACE_CLOSE

#endif

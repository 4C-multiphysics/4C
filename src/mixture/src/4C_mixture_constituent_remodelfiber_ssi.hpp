// This file is part of 4C multiphysics licensed under the
// GNU Lesser General Public License v3.0 or later.
//
// See the LICENSE.md file in the top-level for license information.
//
// SPDX-License-Identifier: LGPL-3.0-or-later

#ifndef FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_SSI_HPP
#define FOUR_C_MIXTURE_CONSTITUENT_REMODELFIBER_SSI_HPP

#include "4C_config.hpp"

#include "4C_io_input_field.hpp"
#include "4C_mat_anisotropy_extension_default.hpp"
#include "4C_mat_fiber_interpolation.hpp"
#include "4C_mat_so3_material.hpp"
#include "4C_material_parameter_base.hpp"
#include "4C_mixture_constituent.hpp"
#include "4C_mixture_constituent_remodelfiber_material.hpp"
#include "4C_mixture_remodelfiber.hpp"

#include <cmath>
#include <cstddef>

FOUR_C_NAMESPACE_OPEN

namespace Mixture
{
  class MixtureConstituent;
  template <typename T>
  class RemodelFiberMaterial;

  namespace PAR
  {
    class MixtureConstituentRemodelFiberSsi : public Mixture::PAR::MixtureConstituent
    {
     public:
      explicit MixtureConstituentRemodelFiberSsi(const Core::Mat::PAR::Parameter::Data& matdata);
      /// create material instance of matching type with my parameters
      std::unique_ptr<Mixture::MixtureConstituent> create_constituent(int id) override;

      const Core::IO::InterpolatedInputField<Core::LinAlg::Tensor<double, 3>,
          Mat::FiberInterpolation>
          fiber_orientation;

      const int fiber_material_id_;
      const Mixture::PAR::RemodelFiberMaterial<double>* fiber_material_;

      const bool enable_growth_;
      const bool enable_basal_mass_production_;
      const double poisson_decay_time_;
      const double growth_constant_;

      const double deposition_stretch_;
      const int deposition_stretch_timefunc_num_;

      const bool inelastic_external_deformation_;

      const std::optional<int> growth_scalar_id_;
      const std::optional<int> remodeling_scalar_id_;
      const std::optional<int> nonlocal_stimulus_scalar_id_;

      const bool implicit_integration_;
    };
  }  // namespace PAR

  /*!
   * \brief Remodel fiber constituent with an explicit update rule
   */
  class MixtureConstituentRemodelFiberSsi : public Mixture::MixtureConstituent
  {
   public:
    explicit MixtureConstituentRemodelFiberSsi(
        Mixture::PAR::MixtureConstituentRemodelFiberSsi* params, int id);

    [[nodiscard]] const Core::LinAlg::SymmetricTensor<double, 3, 3>& get_structural_tensor(
        int gp) const
    {
      return structural_tensors_[gp];
    }

    [[nodiscard]] Core::Materials::MaterialType material_type() const override;

    void pack_constituent(Core::Communication::PackBuffer& data) const override;

    void unpack_constituent(Core::Communication::UnpackBuffer& buffer) override;

    void read_element(int numgp, const Discret::Elements::Fibers& fibers,
        const std::optional<Discret::Elements::CoordinateSystem>& coord_system) override;

    void setup(const Teuchos::ParameterList& params, int eleGID) override;

    void update(const Core::LinAlg::Tensor<double, 3, 3>& F, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context, int gp, int eleGID) override;

    void update_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::Tensor<double, 3, 3>& iFext, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context, double dt, int gp, int eleGID) override;

    void evaluate(const Core::LinAlg::Tensor<double, 3, 3>& F,
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& E, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    void evaluate_elastic_part(const Core::LinAlg::Tensor<double, 3, 3>& FM,
        const Core::LinAlg::Tensor<double, 3, 3>& iFextin, const Teuchos::ParameterList& params,
        const Mat::EvaluationContext<3>& context,
        Core::LinAlg::SymmetricTensor<double, 3, 3>& S_stress,
        Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3>& cmat, int gp, int eleGID) override;

    [[nodiscard]] double get_growth_scalar(int gp) const override;

    [[nodiscard]] double evaluate_growth_reaction_coefficient(int gp) const;
    [[nodiscard]] double evaluate_remodeling_reaction_coefficient(int gp) const;

    [[nodiscard]] double evaluate_local_stimulus(int gp) const;

    [[nodiscard]] bool is_nonlocal_stimulus_mode() const
    {
      return params_->nonlocal_stimulus_scalar_id_.has_value();
    }

    [[nodiscard]] bool is_implicit_integration() const { return params_->implicit_integration_; }

    [[nodiscard]] int nonlocal_stimulus_scalar_id() const
    {
      return params_->nonlocal_stimulus_scalar_id_.value();
    }

    [[nodiscard]] double evaluate_current_fiber_pk2_stress(int gp) const;
    [[nodiscard]] double evaluate_d_growth_scalar_d_nonlocal_stimulus(int gp) const;
    [[nodiscard]] double evaluate_d_cauchy_stress_d_lambda_f_sq(int gp) const;

    void set_current_growth_scalar(const Teuchos::ParameterList& params, int gp);
    void set_current_lambda_r(const Teuchos::ParameterList& params, int gp);

    void register_output_data_names(
        std::unordered_map<std::string, int>& names_and_size) const override;

    bool evaluate_output_data(
        const std::string& name, Core::LinAlg::SerialDenseMatrix& data) const override;

   private:
    [[nodiscard]] double evaluate_lambdaf(
        const Core::LinAlg::SymmetricTensor<double, 3, 3>& C, int gp, int eleGID) const;
    [[nodiscard]] double evaluate_lambda_ext(
        const Core::LinAlg::Tensor<double, 3, 3>& iFext, int gp, int eleGID) const;

    [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3> evaluate_current_pk2(
        int gp, int eleGID) const;
    [[nodiscard]] Core::LinAlg::SymmetricTensor<double, 3, 3, 3, 3> evaluate_current_cmat(
        int gp, int eleGID) const;

    [[nodiscard]] double evaluate_deposition_stretch(double time) const;
    void update_homeostatic_values(
        const Teuchos::ParameterList& params, double total_time, int eleGID);

    void initialize();

    /// my material parameters
    Mixture::PAR::MixtureConstituentRemodelFiberSsi* params_;

    /// An instance of the remodel fiber
    std::vector<RemodelFiber<2>> remodel_fiber_;

    /// Structural tensor of the anisotropy (cached for performance)
    std::vector<Core::LinAlg::SymmetricTensor<double, 3, 3>> structural_tensors_;
  };
}  // namespace Mixture

FOUR_C_NAMESPACE_CLOSE

#endif
